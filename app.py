import os
import json
import datetime
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

# ====== 尝试导入 OpenAI SDK（没装也不要让 app 崩） ====== #
try:
    from openai import OpenAI
except ModuleNotFoundError:
    OpenAI = None

# ===================== 基础配置 ===================== #

st.set_page_config(
    page_title="外卖增长潜力诊断器",
    layout="wide"
)

st.title("📈 餐厅外卖增长潜力诊断器")
st.caption("基于 Google / Yelp / 外卖平台公开页面 + 大模型分析，评估餐厅精细化运营后的外卖增长空间。")

# 行业经验：精细化运营后，正常可提升 15%~60%
MIN_GROWTH = 0.15
MAX_GROWTH = 0.60


# ===================== Secret 读取工具 ===================== #

def get_secret(name: str, default=None):
    """优先从 st.secrets 读取，读取不到则从环境变量中拿。"""
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


YELP_API_KEY = get_secret("YELP_API_KEY")
GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")

if not YELP_API_KEY or not GOOGLE_API_KEY:
    st.warning("⚠️ 未检测到 YELP_API_KEY 或 GOOGLE_API_KEY，请先在 secrets.toml 或环境变量中配置。")

if not OPENAI_API_KEY or OpenAI is None:
    st.info("💡 未配置 OPENAI_API_KEY 或未安装 openai SDK，将跳过 AI 深度分析，只使用规则引擎。")

# OpenAI client（只有 key + SDK 都齐才初始化）
client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_API_KEY and OpenAI is not None) else None


# ===================== Google / Yelp / DuckDuckGo 等调用加缓存 ===================== #

@st.cache_data(show_spinner=False)
def google_geocode(address: str):
    """使用 Google Geocoding API 将地址转为坐标。"""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": GOOGLE_API_KEY}
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    if data.get("status") != "OK":
        return None, None
    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


@st.cache_data(show_spinner=False)
def google_find_place_cached(input_text: str):
    """缓存版 Find Place 基础调用。"""
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "key": GOOGLE_API_KEY,
        "input": input_text,
        "inputtype": "textquery",
        "fields": "place_id,name,geometry,types,rating,user_ratings_total,formatted_address"
    }
    r = requests.get(url, params=params, timeout=15)
    return r.json()


def google_find_place(input_text: str, prefer_restaurant: bool = False, ref_name: str = None):
    """
    使用 Places Find Place API 找到 place 信息。
    - prefer_restaurant=True 时，会优先选餐饮类型 + 有评分的候选
    - ref_name 用于简单名称相似度加权
    """
    data = google_find_place_cached(input_text)
    candidates = data.get("candidates", []) if isinstance(data, dict) else []
    if not candidates:
        return None

    # 不强制餐饮就直接用第一个
    if not prefer_restaurant:
        return candidates[0]

    primary_food_types = {"restaurant", "food", "meal_takeaway", "meal_delivery"}
    secondary_food_types = {"cafe", "bar", "bakery"}

    scored = []
    ref_name_lower = ref_name.lower() if ref_name else None

    for c in candidates:
        score = 0
        types = c.get("types", []) or []

        if any(t in primary_food_types for t in types):
            score += 3
        elif any(t in secondary_food_types for t in types):
            score += 1

        if c.get("user_ratings_total", 0) > 0:
            score += 1

        if ref_name_lower:
            name = c.get("name", "") or ""
            if ref_name_lower in name.lower():
                score += 3

        scored.append((score, c))

    best_score, best_cand = max(scored, key=lambda x: x[0])
    if best_score == 0:
        return candidates[0]
    return best_cand


@st.cache_data(show_spinner=False)
def google_place_details(place_id: str):
    """获取 Place 详情信息（主要用于评分、评论数量）。"""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "name,rating,user_ratings_total,formatted_address,price_level"
    }
    r = requests.get(url, params=params, timeout=15)
    return r.json().get("result", {})


def fetch_google_dinein_menu(address: str) -> pd.DataFrame:
    """
    占位函数：
    Google 官方 API 目前不直接提供结构化菜单。
    如需解析 Google Maps 网页菜单，可在这里扩展 HTML 解析。
    """
    return pd.DataFrame(columns=["name", "price", "category", "channel"])


@st.cache_data(show_spinner=False)
def fetch_yelp_candidates_by_address(address: str, limit: int = 5):
    """
    稳定的地址 → Yelp 匹配。
    """
    if not YELP_API_KEY:
        return []

    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    url = "https://api.yelp.com/v3/businesses/search"

    lat, lng = google_geocode(address)

    if lat is None or lng is None:
        search_params = {"location": address, "limit": limit, "sort_by": "distance"}
    else:
        search_params = {
            "latitude": lat,
            "longitude": lng,
            "radius": 150,   # 单位：米
            "limit": limit,
            "sort_by": "distance"
        }

    r = requests.get(url, headers=headers, params=search_params, timeout=15)
    data = r.json()
    businesses = data.get("businesses", [])
    candidates = []

    if not businesses:
        return []

    for b in businesses:
        display_address = ", ".join(b["location"].get("display_address", []))
        cats = [c["title"] for c in b.get("categories", [])]

        candidates.append(
            {
                "id": b["id"],
                "name": b["name"],
                "rating": b.get("rating", None),
                "review_count": b.get("review_count", 0),
                "price_level": b.get("price", ""),
                "categories": cats,
                "categories_str": ", ".join(cats),
                "lat": b["coordinates"]["latitude"],
                "lng": b["coordinates"]["longitude"],
                "address": display_address,
                "source": "yelp",
            }
        )

    return candidates


@st.cache_data(show_spinner=False)
def fetch_yelp_competitors(lat: float, lng: float, term: str = "", radius_m: int = 600) -> pd.DataFrame:
    """使用 Yelp 搜索附近竞对，默认半径 600m。"""
    if not YELP_API_KEY:
        return pd.DataFrame()

    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    url = "https://api.yelp.com/v3/businesses/search"
    params = {
        "latitude": lat,
        "longitude": lng,
        "radius": radius_m,
        "limit": 10,
        "sort_by": "rating",
    }
    if term:
        params["term"] = term

    r = requests.get(url, headers=headers, params=params, timeout=15)
    data = r.json()
    businesses = data.get("businesses", [])
    if not businesses:
        return pd.DataFrame()

    rows = []
    for b in businesses:
        rows.append({
            "name": b["name"],
            "rating": b.get("rating", None),
            "review_count": b.get("review_count", 0),
            "price_level": b.get("price", ""),
            "distance_m": b.get("distance", None),
            "categories": ", ".join([c["title"] for c in b.get("categories", [])]),
        })
    df = pd.DataFrame(rows)
    if "distance_m" in df.columns:
        df["distance_km"] = df["distance_m"] / 1000.0
    return df


@st.cache_data(show_spinner=False)
def search_duckduckgo(query: str, max_results: int = 3):
    """
    使用 DuckDuckGo 的 HTML 结果页面做简单搜索。
    注意很多链接是 /l/?uddg=，需要解码成真实 URL。
    """
    url = "https://duckduckgo.com/html/"
    params = {"q": query}
    r = requests.get(
        url,
        params=params,
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.select("a.result__a"):
        href = a.get("href")
        if not href:
            continue

        real_url = href
        if href.startswith("/l/"):
            parsed = urlparse(href)
            qs = parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                real_url = unquote(qs["uddg"][0])

        links.append(real_url)
        if len(links) >= max_results:
            break
    return links


def find_delivery_links(restaurant_name: str, address: str):
    """通过搜索找到 Doordash / UberEats 的店铺链接（尽力而为）。"""
    dd_link, ue_link = None, None
    query_base = f'"{restaurant_name}" {address}'

    dd_results = search_duckduckgo(query_base + " site:doordash.com")
    for link in dd_results:
        if "doordash.com" in link:
            dd_link = link
            break

    ue_results = search_duckduckgo(query_base + " site:ubereats.com")
    for link in ue_results:
        if "ubereats.com" in link:
            ue_link = link
            break

    return dd_link, ue_link


@st.cache_data(show_spinner=False)
def parse_doordash_menu(url: str) -> pd.DataFrame:
    """非官方 Doordash 菜单解析，只读公开 HTML。"""
    if not url:
        return pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        items = []
        for block in soup.find_all(["div", "article"]):
            name_tag = block.find("h3")
            if not name_tag:
                continue
            name = name_tag.get_text(strip=True)
            price = None
            for span in block.find_all("span"):
                text = span.get_text(strip=True)
                if text.startswith("$"):
                    try:
                        price = float(text.replace("$", "").strip())
                        break
                    except ValueError:
                        continue
            if name and price is not None:
                items.append({
                    "name": name,
                    "price": price,
                    "category": "Unknown",
                    "channel": "doordash",
                    "tags": []
                })

        return pd.DataFrame(items) if items else pd.DataFrame(
            columns=["name", "price", "category", "channel", "tags"])
    except Exception:
        return pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])


@st.cache_data(show_spinner=False)
def parse_ubereats_menu(url: str) -> pd.DataFrame:
    """非官方 UberEats 菜单解析，只读公开 HTML。"""
    if not url:
        return pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        items = []
        for block in soup.find_all(["div", "article"]):
            name_tag = block.find("h3")
            if not name_tag:
                continue
            name = name_tag.get_text(strip=True)
            price = None
            for span in block.find_all("span"):
                text = span.get_text(strip=True)
                if text.startswith("$"):
                    try:
                        price = float(text.replace("$", "").strip())
                        break
                    except ValueError:
                        continue
            if name and price is not None:
                items.append({
                    "name": name,
                    "price": price,
                    "category": "Unknown",
                    "channel": "ubereats",
                    "tags": []
                })

        return pd.DataFrame(items) if items else pd.DataFrame(
            columns=["name", "price", "category", "channel", "tags"])
    except Exception:
        return pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])


# ===================== 规则评分（六大维度） ===================== #

def compute_menu_structure_score(df_all: pd.DataFrame):
    tips = []
    if df_all is None or df_all.empty:
        return 55.0, ["未成功获取外卖菜单数据，暂用中性偏保守评分。"]

    total_items = len(df_all)
    num_categories = df_all["category"].nunique() if "category" in df_all.columns else 1

    score = 100.0

    if total_items < 10:
        score -= 15
        tips.append(f"当前外卖菜单共 **{total_items}** 个菜品，偏少，用户选择有限，建议补充 2–3 个高毛利 Star Item。")
    elif total_items > 60:
        score -= 25
        tips.append(f"当前外卖菜单共 **{total_items}** 个菜品，>60 个，容易导致选择困难，建议精简和合并部分菜品。")
    else:
        tips.append(f"当前外卖菜单单品数量约 **{total_items}** 个，处在可控范围内。")

    if num_categories > 8:
        score -= 15
        tips.append(f"菜单类别数量约 **{num_categories}** 个，偏多，建议压缩到 5–7 个主类目，突出主力品类。")
    else:
        tips.append(f"菜单类别数量约 **{num_categories}** 个。")

    has_combo = False
    for c in df_all.get("category", pd.Series()).dropna().astype(str):
        if "combo" in c.lower() or "套餐" in c:
            has_combo = True
            break
    if not has_combo:
        score -= 10
        tips.append("缺少套餐/组合菜单，建议设计 2–3 个客单价更高的套餐组合，提升客单价。")
    else:
        tips.append("已检测到套餐/组合类目，可在此基础上继续优化客单价结构。")

    return max(score, 0), tips


def compute_pricing_score(df_dinein: pd.DataFrame, df_delivery: pd.DataFrame):
    tips = []
    if df_dinein is None or df_dinein.empty or df_delivery is None or df_delivery.empty:
        return 60.0, ["缺少完整的堂食/外卖价格对比数据，暂用中性评分。"]

    merge = pd.merge(
        df_dinein[["name", "price"]],
        df_delivery[["name", "price"]],
        on="name",
        suffixes=("_dinein", "_delivery")
    )
    if merge.empty:
        return 60.0, ["堂食与外卖未找到重叠菜品，无法精确比较加价率。"]

    merge["markup"] = (merge["price_delivery"] - merge["price_dinein"]) / merge["price_dinein"]
    avg_markup = merge["markup"].mean()

    score = 100.0
    if avg_markup < 0.10:
        score -= 15
        tips.append(f"当前可比菜品平均外卖加价率约 **{avg_markup:.0%}**，偏低，建议适当提高到 15% 左右以覆盖平台与配送成本。")
    elif avg_markup > 0.35:
        score -= 20
        tips.append(f"当前可比菜品平均外卖加价率约 **{avg_markup:.0%}**，偏高，可能影响转化率，建议控制在 15%–30% 区间。")
    else:
        tips.append(f"当前可比菜品平均外卖加价率约 **{avg_markup:.0%}**，整体合理。")

    tips.append(f"用于分析的可比菜品数量：**{len(merge)}** 个。")
    return max(score, 0), tips


def compute_promotion_score(has_dd_link: bool, has_ue_link: bool):
    tips = []
    if not has_dd_link and not has_ue_link:
        score = 45.0
        tips.append("暂未发现 Doordash / UberEats 店铺链接，外卖渠道基础需要先补齐。")
    elif has_dd_link and has_ue_link:
        score = 70.0
        tips.append("已覆盖主流外卖平台，适合做分平台差异化优惠与老客复购活动。")
    else:
        score = 60.0
        tips.append("外卖平台仅覆盖部分渠道，建议同步拓展至 Doordash + UberEats，并统一价格与活动策略。")

    tips.append("当前版本未读取具体活动内容，建议后续落地：首单减免、午晚高峰满减、老客券包等组合玩法，把一次性流量变成可复购用户。")
    return score, tips


def compute_competitor_score(df_comp: pd.DataFrame, restaurant_rating: float):
    tips = []
    if df_comp is None or df_comp.empty or restaurant_rating is None:
        return 60.0, ["竞对或本店评分数据不完整，暂用中性评分。"]

    avg_comp_rating = df_comp["rating"].mean()
    diff = restaurant_rating - avg_comp_rating
    score = 60.0 + diff * 10
    score = max(min(score, 100.0), 0.0)

    tips.append(
        f"附近 600m 内共检测到 **{len(df_comp)}** 家同类竞对门店，平均评分约 **{avg_comp_rating:.1f}** 分。"
    )
    if diff >= 0.2:
        tips.append(f"本店 Yelp 评分 **{restaurant_rating:.1f}**，高于竞对均值 {avg_comp_rating:.1f}，口碑具备优势，可以在外卖详情页更突出。")
    elif diff <= -0.2:
        tips.append(f"本店 Yelp 评分 **{restaurant_rating:.1f}**，低于竞对均值 {avg_comp_rating:.1f}，建议通过服务体验、包装、好评激励活动快速拉升评分。")
    else:
        tips.append("本店评分与附近竞对大致持平，建议通过菜品照片、文案与活动玩法做差异化。")

    if "distance_km" in df_comp.columns and not df_comp["distance_km"].isna().all():
        tips.append(
            f"已检测到的竞对距离本店约 **{df_comp['distance_km'].min():.2f}–{df_comp['distance_km'].max():.2f} km**，"
            "意味着用户在同一配送半径内有多家可选。"
        )

    return score, tips


def compute_coverage_score():
    score = 70.0
    tips = [
        "从地理位置和商圈结构的通用经验看，配送覆盖具备一定潜力，"
        "后续可结合实际平台配送半径与学校/写字楼密度做进一步量化。"
    ]
    return score, tips


def compute_market_voice_score(yelp_info: dict, place_info: dict):
    """
    市场声音（0–100）：综合 Yelp + Google 的评分 & 评论量。
    """
    tips = []

    y_rating = yelp_info.get("rating")
    y_reviews = yelp_info.get("review_count", 0)

    g_rating = None
    g_reviews = 0
    if place_info:
        g_rating = place_info.get("rating")
        g_reviews = place_info.get("user_ratings_total", 0)

    score = 60.0

    if y_rating is not None:
        score += (y_rating - 4.0) * 5
        tips.append(f"Yelp 评分：**{y_rating:.1f}** 分，评论数约 **{y_reviews}** 条。")
    else:
        tips.append("Yelp 暂无评分数据。")

    if g_rating is not None:
        score += (g_rating - 4.0) * 5
        tips.append(f"Google 评分：**{g_rating:.1f}** 分，评论数约 **{g_reviews}** 条。")
    else:
        tips.append("Google 暂无评分数据或未收录。")

    total_reviews = (y_reviews or 0) + (g_reviews or 0)
    if total_reviews < 50:
        score -= 5
        tips.append("总体线上评论量偏少，市场声音相对有限，可通过引导好评、做活动提升评论基数。")
    elif total_reviews > 300:
        score += 5
        tips.append("总体线上评论量较多，品牌在本地有一定“存在感”，可以放大复购与口碑转介绍。")

    score = max(min(score, 100.0), 0.0)
    tips.append("当前版本暂未接入外卖平台（Doordash/UberEats）的独立评分，仅基于 Yelp + Google 做统一评估。")

    return score, tips


def compute_growth_rate(menu_score, price_score, promo_score, comp_score, coverage_score, voice_score) -> float:
    """六大维度加权，映射到 15%~60% 的增长区间。"""
    weighted = (
        0.18 * menu_score +
        0.12 * price_score +
        0.20 * promo_score +
        0.12 * comp_score +
        0.18 * coverage_score +
        0.20 * voice_score
    ) / 100.0
    growth_rate = MIN_GROWTH + (MAX_GROWTH - MIN_GROWTH) * weighted
    return growth_rate


# ===================== 标准化 Schema + LLM 分析 ===================== #

def build_standard_payload(address: str, result: dict) -> dict:
    """把规则引擎的 result 映射成标准化 JSON 结构，喂给大模型。"""
    yi = result.get("yelp_info", {}) or {}
    gi = result.get("place_info", {}) or {}
    menus = result.get("menus", {}) or {}
    all_df = menus.get("all")
    comp_df = result.get("competitors")

    # 菜单统计
    total_items = int(len(all_df)) if all_df is not None else 0
    num_categories = 0
    if all_df is not None and "category" in all_df.columns:
        num_categories = int(all_df["category"].nunique())

    prices = []
    if all_df is not None and "price" in all_df.columns:
        prices = [p for p in all_df["price"].tolist() if isinstance(p, (int, float))]
    if prices:
        min_price = float(min(prices))
        max_price = float(max(prices))
        median_price = float(np.median(prices))
    else:
        min_price = max_price = median_price = None

    # 竞对统计
    num_competitors = 0
    avg_comp_rating = None
    med_comp_rating = None
    if comp_df is not None and not comp_df.empty and "rating" in comp_df.columns:
        num_competitors = int(len(comp_df))
        avg_comp_rating = float(comp_df["rating"].mean())
        med_comp_rating = float(comp_df["rating"].median())

    payload = {
        "meta": {
            "version": "restaurant_schema_v1",
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z"
        },
        "basic_info": {
            "name": yi.get("name") or gi.get("name"),
            "address": yi.get("address") or gi.get("formatted_address") or address,
            "lat": yi.get("lat"),
            "lng": yi.get("lng")
        },
        "online_presence": {
            "yelp": {
                "rating": yi.get("rating"),
                "review_count": yi.get("review_count", 0),
                "price_level": yi.get("price_level"),
                "categories": yi.get("categories", [])
            },
            "google": {
                "rating": gi.get("rating"),
                "review_count": gi.get("user_ratings_total", 0)
            }
        },
        "delivery_channels": {
            "doordash": {
                "has_link": bool(result["delivery_links"].get("doordash")),
                "url": result["delivery_links"].get("doordash")
            },
            "ubereats": {
                "has_link": bool(result["delivery_links"].get("ubereats")),
                "url": result["delivery_links"].get("ubereats")
            }
        },
        "menu": {
            "total_items": total_items,
            "num_categories": num_categories,
            "channels": {
                "dinein_items": int(len(menus.get("dinein"))) if menus.get("dinein") is not None else 0,
                "doordash_items": int(len(menus.get("doordash"))) if menus.get("doordash") is not None else 0,
                "ubereats_items": int(len(menus.get("ubereats"))) if menus.get("ubereats") is not None else 0
            },
            "price_summary": {
                "min_price": min_price,
                "max_price": max_price,
                "median_price": median_price
            }
        },
        "competition": {
            "num_competitors": num_competitors,
            "avg_rating": avg_comp_rating,
            "median_rating": med_comp_rating
        },
        "scores": result.get("scores", {}),
        "growth_estimation": {
            "growth_rate": float(result.get("growth_rate", 0.0)),
            "current_daily_revenue": float(result.get("current_daily_revenue", 0.0)),
            "potential_daily_revenue": float(result.get("potential_daily_revenue", 0.0))
        }
    }
    return payload


def llm_deep_analysis(payload: dict) -> dict:
    """
    使用 GPT-5 Responses API 进行深度分析。
    新版 API 不支持 response_format，只能在 prompt 里强制模型输出 JSON。
    """
    if client is None:
        return {
            "overall_summary": "未配置 OPENAI_API_KEY 或未安装 openai SDK，当前仅展示规则引擎结果，未启用 AI 深度分析。",
            "key_findings": [],
            "prioritized_actions": [],
            "risks": [],
            "data_gaps": []
        }

    prompt = f"""
你是一名北美餐饮 & 外卖运营专家，熟悉 DoorDash、UberEats、中餐厅经营。

下面是该餐厅的结构化 JSON 信息：
{json.dumps(payload, ensure_ascii=False, indent=2)}

请基于以上信息输出餐厅的深度诊断结果。

⚠️ 输出格式必须是 严格 JSON，不要出现多余文字、不允许加解释、不允许 Markdown。

固定输出 JSON schema 如下：

{{
  "overall_summary": "string，整体总结",
  "key_findings": ["string 列表，核心洞察"],
  "prioritized_actions": [
    {{
      "horizon": "short_term 或 mid_term",
      "description": "行动建议"
    }}
  ],
  "risks": ["string 列表，主要风险点"],
  "data_gaps": ["string 列表，缺失的数据"]
}}

只返回 JSON，不能出现代码块、注释、额外说明。
"""

    resp = client.responses.create(
        model="gpt-5.1-mini",
        input=prompt,
        max_output_tokens=1500
    )

    raw_output = resp.output_text

    # 尝试解析 JSON
    try:
        return json.loads(raw_output)
    except Exception:
        try:
            fixed = raw_output[raw_output.find("{"): raw_output.rfind("}") + 1]
            return json.loads(fixed)
        except Exception:
            return {
                "overall_summary": raw_output,
                "key_findings": [],
                "prioritized_actions": [],
                "risks": [],
                "data_gaps": []
            }


@st.cache_data(show_spinner=False)
def llm_deep_analysis_cached(payload_json_str: str) -> dict:
    """带缓存的 LLM 调用，避免同一餐厅反复扣费+等待。"""
    payload = json.loads(payload_json_str)
    return llm_deep_analysis(payload)


# ===================== 核心分析管线 ===================== #

def analyze_restaurant(address: str, avg_orders: float, avg_ticket: float,
                       yelp_business: dict, fast_mode: bool = False):
    """
    主入口：
    - address：用户输入的地址
    - yelp_business：用户在候选列表中选择的店
    - fast_mode：True 时跳过菜单抓取和 LLM，只用于规则层面快速评估
    """
    if not yelp_business:
        raise RuntimeError("未提供有效的 Yelp / Google 店铺信息。")

    yelp_info = yelp_business

    # 让 Google 更精确：店名 + 地址
    place_query = f"{yelp_info['name']} {address}"
    place_info = None
    place = google_find_place(place_query, prefer_restaurant=True, ref_name=yelp_info["name"])
    if place and place.get("place_id"):
        place_info = google_place_details(place["place_id"])

    # 竞对（Yelp）
    comp_df = fetch_yelp_competitors(yelp_info["lat"], yelp_info["lng"])

    # 菜单 & 外卖渠道
    dinein_df = fetch_google_dinein_menu(address)  # 目前为空占位

    if fast_mode:
        dd_link = ue_link = None
        dd_df = ue_df = pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])
    else:
        dd_link, ue_link = find_delivery_links(yelp_info["name"], yelp_info["address"])
        dd_df = parse_doordash_menu(dd_link)
        ue_df = parse_ubereats_menu(ue_link)

    if dd_df.empty and ue_df.empty:
        all_df = pd.DataFrame()
    else:
        all_df = pd.concat([dinein_df, dd_df, ue_df], ignore_index=True)

    # 六大维度
    menu_score, menu_tips = compute_menu_structure_score(all_df)
    price_score, price_tips = compute_pricing_score(dinein_df, dd_df if not dd_df.empty else ue_df)
    promo_score, promo_tips = compute_promotion_score(
        has_dd_link=dd_link is not None, has_ue_link=ue_link is not None
    )
    comp_score, comp_tips = compute_competitor_score(comp_df, yelp_info.get("rating", None))
    coverage_score, coverage_tips = compute_coverage_score()
    voice_score, voice_tips = compute_market_voice_score(yelp_info, place_info)

    growth_rate = compute_growth_rate(
        menu_score, price_score, promo_score, comp_score, coverage_score, voice_score
    )

    current_daily_revenue = avg_orders * avg_ticket
    potential_daily_revenue = current_daily_revenue * (1 + growth_rate)
    revenue_uplift_daily = potential_daily_revenue - current_daily_revenue
    revenue_uplift_monthly = revenue_uplift_daily * 30

    result = {
        "yelp_info": yelp_info,
        "place_info": place_info,
        "competitors": comp_df,
        "delivery_links": {
            "doordash": dd_link,
            "ubereats": ue_link
        },
        "menus": {
            "dinein": dinein_df,
            "doordash": dd_df,
            "ubereats": ue_df,
            "all": all_df
        },
        "scores": {
            "菜单结构": menu_score,
            "定价与客单价": price_score,
            "活动体系": promo_score,
            "竞对压力": comp_score,
            "覆盖与圈层": coverage_score,
            "市场声音": voice_score,
        },
        "tips": {
            "菜单结构": menu_tips,
            "定价与客单价": price_tips,
            "活动体系": promo_tips,
            "竞对压力": comp_tips,
            "覆盖与圈层": coverage_tips,
            "市场声音": voice_tips,
        },
        "growth_rate": growth_rate,
        "current_daily_revenue": current_daily_revenue,
        "potential_daily_revenue": potential_daily_revenue,
        "revenue_uplift_daily": revenue_uplift_daily,
        "revenue_uplift_monthly": revenue_uplift_monthly,
        "fast_mode": fast_mode,
    }

    return result


# ===================== Streamlit 状态初始化 ===================== #

if "yelp_candidates" not in st.session_state:
    st.session_state["yelp_candidates"] = []
if "selected_yelp_index" not in st.session_state:
    st.session_state["selected_yelp_index"] = None
if "confirmed_address" not in st.session_state:
    st.session_state["confirmed_address"] = ""


# ===================== UI：第一步 地址输入 + 匹配餐厅 ===================== #

st.subheader("📍 第一步：输入地址并匹配餐厅")

with st.form("address_form"):
    raw_address = st.text_input(
        "餐厅地址（用于匹配 Yelp / Google / 外卖平台）",
        value=st.session_state.get("confirmed_address", "")
    )
    match_submitted = st.form_submit_button("🔍 匹配该地址下的餐厅")

if match_submitted:
    if not raw_address.strip():
        st.error("请输入餐厅地址。")
    else:
        with st.spinner("正在根据地址匹配 Yelp 餐厅，请稍等..."):
            candidates = fetch_yelp_candidates_by_address(raw_address)

        if not candidates:
            place = google_find_place(raw_address, prefer_restaurant=True)
            if place:
                types = place.get("types", []) or []

                primary_food_types = {"restaurant", "food", "meal_takeaway", "meal_delivery"}
                secondary_food_types = {"cafe", "bar", "bakery"}

                is_primary = any(t in primary_food_types for t in types)
                is_secondary = any(t in secondary_food_types for t in types)

                if is_primary or is_secondary:
                    details = google_place_details(place["place_id"])
                    loc = place["geometry"]["location"]

                    google_candidate = {
                        "id": None,
                        "name": details.get("name", place.get("name", "Unknown Business")),
                        "rating": details.get("rating", None),
                        "review_count": details.get("user_ratings_total", 0),
                        "price_level": details.get("price_level", ""),
                        "categories": types,
                        "categories_str": ", ".join(types) if types else "Google Place",
                        "lat": loc["lat"],
                        "lng": loc["lng"],
                        "address": details.get("formatted_address", raw_address),
                        "source": "google",
                    }
                    candidates = [google_candidate]

        st.session_state["confirmed_address"] = raw_address
        st.session_state["yelp_candidates"] = candidates
        st.session_state["selected_yelp_index"] = 0 if candidates else None

candidates = st.session_state.get("yelp_candidates", [])
selected_biz = None

if candidates:
    st.success("已在该地址附近匹配到以下餐厅，请选择要诊断的一家：")

    options = list(range(len(candidates)))

    def format_option(i):
        b = candidates[i]
        source = b.get("source", "yelp")
        source_tag = "Yelp" if source == "yelp" else "Google"
        return f"{b['name']} · {b['categories_str']} · ⭐ {b.get('rating', 'N/A')} · {b['address']} · {source_tag}"

    selected_index = st.radio(
        "匹配餐厅",
        options,
        format_func=format_option,
        index=st.session_state.get("selected_yelp_index", 0)
    )
    st.session_state["selected_yelp_index"] = selected_index
    selected_biz = candidates[selected_index]

    st.info(
        f"当前已选择：**{selected_biz['name']}**（{selected_biz['address']}）。"
        "点击下方“开始诊断”前，可以先确认是否是你要分析的那家店。"
    )

elif st.session_state["confirmed_address"]:
    st.error("该地址附近未在 Yelp / Google 找到餐厅业务，可能不是餐厅地址或未登记为餐饮门店。")


# ===================== UI：第二步 输入业务数据 + 开始诊断 ===================== #

st.subheader("📊 第二步：输入当前外卖数据，生成诊断结果")

with st.form("diagnose_form"):
    col1, col2 = st.columns(2)
    with col1:
        avg_orders = st.number_input("当前日均外卖单量（单）", min_value=0.0, value=40.0, step=1.0)
    with col2:
        avg_ticket = st.number_input("当前外卖客单价（美元）", min_value=0.0, value=25.0, step=1.0)

    fast_mode = st.checkbox("⚡ 快速诊断模式（跳过菜单抓取 & 大模型分析，提升速度）", value=True)

    start_diagnose = st.form_submit_button("🚀 开始诊断")

if start_diagnose:
    if not selected_biz:
        st.error("请先在上方匹配并选择一家餐厅。当前地址可能不是餐厅，或者 Yelp / Google 上没有相关店铺。")
    else:
        try:
            with st.spinner("正在基于 Yelp / Google / 外卖平台数据进行诊断..."):
                result = analyze_restaurant(
                    st.session_state["confirmed_address"],
                    avg_orders,
                    avg_ticket,
                    yelp_business=selected_biz,
                    fast_mode=fast_mode,
                )

            # 顶部 KPI
            st.subheader("📌 诊断结果总览")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("当前日外卖营业额（估算）", f"${result['current_daily_revenue']:.0f}")
            with col_b:
                st.metric("优化后日外卖营业额（预测）", f"${result['potential_daily_revenue']:.0f}")
            with col_c:
                st.metric("月度可提升外卖营业额（预测）", f"+${result['revenue_uplift_monthly']:.0f}")

            st.write(
                f"综合菜单结构、定价策略、活动体系、竞对压力、覆盖圈层与市场声音，"
                f"系统预估通过精细化运营，可带来约 **{result['growth_rate']*100:.1f}%** 的外卖营业额增长空间。"
            )

            # 六大维度评分
            st.subheader("🧬 六大维度诊断评分")
            score_df = pd.DataFrame(
                {"维度": list(result["scores"].keys()),
                 "得分": list(result["scores"].values())}
            )
            st.bar_chart(score_df.set_index("维度"))

            # ===== AI 深度诊断（仅在非快速模式下启用） =====
            if not fast_mode:
                payload = build_standard_payload(
                    st.session_state["confirmed_address"],
                    result
                )
                payload_str = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                with st.spinner("正在调用 AI 进行深度逻辑分析..."):
                    ai_analysis = llm_deep_analysis_cached(payload_str)

                st.subheader("🧠 AI 深度诊断（大模型分析）")
                tab_ai, tab_data = st.tabs(["AI 诊断结论", "特征 JSON"])

                with tab_ai:
                    st.markdown(f"**整体总结：** {ai_analysis.get('overall_summary', '')}")

                    st.markdown("**关键发现：**")
                    for item in ai_analysis.get("key_findings", []):
                        st.markdown(f"- {item}")

                    st.markdown("**优先行动清单：**")
                    for act in ai_analysis.get("prioritized_actions", []):
                        st.markdown(f"- [{act.get('horizon', 'short_term')}] {act.get('description','')}")

                    st.markdown("**潜在风险点：**")
                    for r in ai_analysis.get("risks", []):
                        st.markdown(f"- {r}")

                    st.markdown("**数据缺口（建议补充）：**")
                    for g in ai_analysis.get("data_gaps", []):
                        st.markdown(f"- {g}")

                with tab_data:
                    st.code(json.dumps(payload, indent=2, ensure_ascii=False), language="json")
            else:
                st.info("当前为⚡快速诊断模式：已跳过菜单抓取与大模型分析，只展示基础评分与竞对。")

            # 分维度建议 + 具体分析
            st.subheader("🩺 分维度运营建议（点击展开查看详细分析）")
            for dim, tips in result["tips"].items():
                with st.expander(f"{dim} · 诊断与分析概览"):
                    for t in tips:
                        st.markdown(f"- {t}")

                    if dim == "竞对压力":
                        comp_df = result["competitors"]
                        if comp_df is not None and not comp_df.empty:
                            st.markdown("**附近竞对概览：**")
                            st.write(
                                f"- 竞对数量：**{len(comp_df)}** 家\n"
                                f"- 评分中位数：**{comp_df['rating'].median():.1f}**\n"
                            )
                            if "distance_km" in comp_df.columns:
                                st.write(
                                    f"- 距离范围：约 **{comp_df['distance_km'].min():.2f}–{comp_df['distance_km'].max():.2f} km**"
                                )
                            st.markdown("**评分最高的前 5 家竞对：**")
                            st.dataframe(
                                comp_df.sort_values("rating", ascending=False)
                                .head(5)[["name", "rating", "review_count", "price_level", "distance_km", "categories"]]
                            )
                        else:
                            st.write("未获取到竞对数据。")

                    if dim == "菜单结构":
                        all_df = result["menus"]["all"]
                        if all_df is not None and not all_df.empty:
                            st.markdown("**菜单结构概览：**")
                            st.write(f"- 外卖 & 堂食合计菜品数：**{len(all_df)}** 个")
                            if "category" in all_df.columns:
                                st.write(f"- 类目数量：**{all_df['category'].nunique()}** 个")
                                st.markdown("**各类目菜品数 Top5：**")
                                st.dataframe(
                                    all_df.groupby("category")["name"]
                                    .count()
                                    .sort_values(ascending=False)
                                    .head(5)
                                    .rename("菜品数")
                                )
                        else:
                            st.write("未获取到菜单结构数据。")

                    if dim == "市场声音":
                        yi = result["yelp_info"]
                        gi = result.get("place_info")
                        st.markdown("**线上口碑总览：**")
                        if yi:
                            st.write(
                                f"- Yelp：评分 **{yi.get('rating', 'N/A')}**，评论 **{yi.get('review_count', 0)}** 条"
                            )
                        if gi:
                            st.write(
                                f"- Google：评分 **{gi.get('rating', 'N/A')}**，评论 **{gi.get('user_ratings_total', 0)}** 条"
                            )

                        total_reviews = (yi.get("review_count", 0) if yi else 0) + (
                            gi.get("user_ratings_total", 0) if gi else 0
                        )
                        st.write(f"- Yelp + Google 总评论量约：**{total_reviews}** 条")

                        st.markdown(
                            "**策略建议：** 可以在门店桌牌、收据、外卖贴纸上做“好评返券/积分”活动，"
                            "让评论量更快破 300 以上，把“市场声音”做成真实的投放资产。"
                        )

            # 基本信息
            st.subheader("🏪 店铺基础信息（来自 Yelp / Google）")
            col_y1, col_y2 = st.columns(2)
            with col_y1:
                st.markdown("**Yelp 信息**")
                yi = result["yelp_info"]
                st.write(f"店名：{yi['name']}")
                st.write(f"地址：{yi['address']}")
                st.write(f"评分：{yi.get('rating', 'N/A')} ⭐️（{yi.get('review_count', 0)} 条评论）")
                st.write(f"价格等级：{yi.get('price_level', 'N/A')}")
                st.write(f"品类：{', '.join(yi.get('categories', []))}")
            with col_y2:
                st.markdown("**Google Place 信息（若匹配成功）**")
                gi = result.get("place_info")
                if gi:
                    st.write(f"店名：{gi.get('name', 'N/A')}")
                    st.write(f"地址：{gi.get('formatted_address', 'N/A')}")
                    st.write(f"评分：{gi.get('rating', 'N/A')} ⭐️（{gi.get('user_ratings_total', 0)} 条评论）")
                else:
                    st.write("未从 Google Places 找到更多详情。")

            # 外卖平台链接
            st.subheader("🚚 外卖平台覆盖情况")
            dl = result["delivery_links"]
            if dl["doordash"]:
                st.markdown(f"- ✅ Doordash：[{dl['doordash']}]({dl['doordash']})")
            else:
                st.markdown("- ❌ 未发现 Doordash 店铺链接")
            if dl["ubereats"]:
                st.markdown(f"- ✅ UberEats：[{dl['ubereats']}]({dl['ubereats']})")
            else:
                st.markdown("- ❌ 未发现 UberEats 店铺链接")

            # 菜单数据
            st.subheader("📑 菜单数据（若解析成功）")
            tab1, tab2, tab3, tab4 = st.tabs(["堂食（Google）", "Doordash 菜单", "UberEats 菜单", "整合视图"])
            with tab1:
                if result["menus"]["dinein"].empty:
                    st.write("当前版本未从 Google 解析结构化堂食菜单。")
                else:
                    st.dataframe(result["menus"]["dinein"])
            with tab2:
                if result["menus"]["doordash"].empty:
                    st.write("未解析到 Doordash 菜单结构。")
                else:
                    st.dataframe(result["menus"]["doordash"])
            with tab3:
                if result["menus"]["ubereats"].empty:
                    st.write("未解析到 UberEats 菜单结构。")
                else:
                    st.dataframe(result["menus"]["ubereats"])
            with tab4:
                if result["menus"]["all"].empty:
                    st.write("暂无可用菜单数据。")
                else:
                    st.dataframe(result["menus"]["all"])

            # 竞对列表
            st.subheader("🏁 附近竞对门店列表（来自 Yelp）")
            if result["competitors"].empty:
                st.write("未获取到竞对数据。")
            else:
                st.dataframe(result["competitors"])

            st.info(
                "当前版本：先由 Yelp + 坐标匹配餐厅，若失败则由 Google Places 兜底；"
                "只有当 Yelp 和 Google 都无法识别为餐饮门店时，才会提示“该地址不是餐厅”。"
            )

        except Exception as e:
            st.error(f"诊断过程中出现错误：{e}")

else:
    st.markdown(
        """
        ### 使用说明
        1. 在上方输入餐厅地址并点击「匹配该地址下的餐厅」  
        2. 从候选列表中选中你的餐厅  
        3. 输入当前日均外卖单量 & 客单价，选择是否开启⚡快速诊断模式，点击「开始诊断」  
        4. 快速模式下只看基础盘子；关闭快速模式则会抓菜单 + 调大模型出一份深度报告  
        """
    )

# ========== 署名（LinkedIn） ==========
LINKEDIN_URL = "https://www.linkedin.com/in/lingyu-maxwell-lai"
st.markdown(
    f"""
<div style="display:flex;align-items:center;gap:10px;margin-top:-6px;margin-bottom:8px;">
  <div style="font-size:14px;color:#666;">
    Builded by <strong>Maxwell Lai</strong>
  </div>
  <a href="{LINKEDIN_URL}" target="_blank" title="LinkedIn: Maxwell Lai"
     style="display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;
            border-radius:4px;background:#0A66C2;">
    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg"
         alt="LinkedIn" width="12" height="12" style="filter: invert(1);" />
  </a>
</div>
""",
    unsafe_allow_html=True,
)
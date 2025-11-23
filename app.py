import os
import requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup

# ===================== 基础配置 ===================== #

st.set_page_config(
    page_title="外卖增长潜力诊断器",
    layout="wide"
)

st.title("📈 餐厅外卖增长潜力诊断器")
st.caption("基于 Google / Yelp / 外卖平台公开页面，评估餐厅精细化运营后的外卖增长空间。")

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

if not YELP_API_KEY or not GOOGLE_API_KEY:
    st.warning("⚠️ 未检测到 YELP_API_KEY 或 GOOGLE_API_KEY，请先在 secrets.toml 或环境变量中配置。")


# ===================== Google API 相关 ===================== #

def google_geocode(address: str):
    """使用 Google Geocoding API 将地址转为坐标。"""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": address,
        "key": GOOGLE_API_KEY
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    if data.get("status") != "OK":
        return None, None
    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


def google_find_place(address: str):
    """使用 Places Find Place API 找到 place_id + types。"""
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "key": GOOGLE_API_KEY,
        "input": address,
        "inputtype": "textquery",
        "fields": "place_id,name,geometry,types"
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return None
    return candidates[0]


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


# ===================== Yelp API：地址 → 餐厅候选 ===================== #

def is_restaurant_business(biz: dict) -> bool:
    """根据 Yelp 类别判断是否为餐厅/食品相关。"""
    cats = biz.get("categories", []) or []
    aliases = [c.get("alias", "").lower() for c in cats]
    titles = [c.get("title", "").lower() for c in cats]

    if any(a in ("restaurants", "food") for a in aliases):
        return True
    if any("restaurant" in t or "餐厅" in t for t in titles):
        return True
    return False


def fetch_yelp_candidates_by_address(address: str, limit: int = 5):
    """
    用地址在 Yelp 搜索出“这个地址附近的餐厅列表”，
    相当于后端的地址自动补全 + 店铺确认。
    """
    if not YELP_API_KEY:
        return []

    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    url = "https://api.yelp.com/v3/businesses/search"
    params = {
        "location": address,
        "limit": limit,
        "sort_by": "distance"
    }

    r = requests.get(url, headers=headers, params=params, timeout=15)
    data = r.json()
    businesses = data.get("businesses", [])
    candidates = []

    for b in businesses:
        if not is_restaurant_business(b):
            continue
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
                "source": "yelp",   # 标记是 Yelp 来的
            }
        )

    return candidates


def fetch_yelp_competitors(lat: float, lng: float, term: str = "", radius_m: int = 1000) -> pd.DataFrame:
    """使用 Yelp 搜索 1km 内竞对。"""
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


# ===================== 外卖平台搜索 & 菜单解析 ===================== #

def search_duckduckgo(query: str, max_results: int = 5):
    """使用 DuckDuckGo 的 HTML 结果页面做简单搜索。"""
    url = "https://duckduckgo.com/html/"
    params = {"q": query}
    r = requests.get(url, params=params, timeout=15,
                     headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    links = []
    for a in soup.select("a.result__a"):
        href = a.get("href")
        if href:
            links.append(href)
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


def parse_doordash_menu(url: str) -> pd.DataFrame:
    """非官方 Doordash 菜单解析，只读公开 HTML。"""
    if not url:
        return pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=20).text
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
            if name and price:
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


def parse_ubereats_menu(url: str) -> pd.DataFrame:
    """非官方 UberEats 菜单解析，只读公开 HTML。"""
    if not url:
        return pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=20).text
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
            if name and price:
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


# ===================== 分析逻辑（六大维度） ===================== #

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
        f"附近 1km 内共检测到 **{len(df_comp)}** 家同类竞对门店，平均评分约 **{avg_comp_rating:.1f}** 分。"
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
    新增指标：市场声音（0–100）
    综合 Yelp + Google 的评分 & 评论量。
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
    """
    六大维度加权。
    """
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


# ===================== 核心分析管线 ===================== #

def analyze_restaurant(address: str, avg_orders: float, avg_ticket: float, yelp_business: dict):
    """
    入口：
    - address：用户输入的地址（方便 Google）
    - yelp_business：用户在候选列表中选择的那家店
    """
    if not yelp_business:
        raise RuntimeError("未提供有效的 Yelp / Google 店铺信息。")

    yelp_info = yelp_business

    # Google Place 信息
    place_info = None
    place = google_find_place(address)
    if place and place.get("place_id"):
        place_info = google_place_details(place["place_id"])

    # 竞对（按 Yelp）
    comp_df = fetch_yelp_competitors(yelp_info["lat"], yelp_info["lng"])

    # 菜单数据
    dinein_df = fetch_google_dinein_menu(address)  # 当前为空占位
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
    }

    return result


# ===================== Streamlit 状态初始化 ===================== #

if "yelp_candidates" not in st.session_state:
    st.session_state["yelp_candidates"] = []
if "selected_yelp_index" not in st.session_state:
    st.session_state["selected_yelp_index"] = None
if "confirmed_address" not in st.session_state:
    st.session_state["confirmed_address"] = ""


# ===================== UI：第一步 地址输入 + 自动匹配餐厅 ===================== #

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
        # 先用 Yelp 找附近餐厅
        with st.spinner("正在根据地址匹配 Yelp 餐厅，请稍等..."):
            candidates = fetch_yelp_candidates_by_address(raw_address)

        # 如果 Yelp 没找到，再用 Google Places 兜底
        if not candidates:
            place = google_find_place(raw_address)
            if place:
                types = place.get("types", []) or []

                # 严格的餐厅类型判断
                primary_food_types = {
                    "restaurant",
                    "food",
                    "meal_takeaway",
                    "meal_delivery",
                }
                secondary_food_types = {"cafe", "bar", "bakery"}

                is_primary = any(t in primary_food_types for t in types)
                is_secondary = any(t in secondary_food_types for t in types)

                if is_primary:
                    # 用 Google 详情补全信息
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
                        "source": "google",  # 标记来源是 Google
                    }
                    candidates = [google_candidate]
                elif is_secondary:
                    # 弱餐饮类型（cafe/bar等），保守起见也可以给出，让用户判断要不要用
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
                        "source": "google",  # 依然标记 Google
                    }
                    candidates = [google_candidate]
                # 否则：Google 也认为不是餐饮相关，就保持 candidates 为空

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
                    yelp_business=selected_biz
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

            # 维度建议 + 具体分析
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
                            st.write("未获取到有效竞对数据。")

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
                "当前版本：先由 Yelp 匹配餐厅，若失败则由 Google Places 兜底；"
                "只有当 Yelp 和 Google 都无法识别为餐厅类型时，才会提示“该地址不是餐厅”。"
            )

        except Exception as e:
            st.error(f"诊断过程中出现错误：{e}")

else:
    st.markdown(
        """
        ### 使用说明
        1. **先在上方输入地址并点击「匹配该地址下的餐厅」**：  
           - 系统先用 Yelp 搜索附近餐厅；  
           - 若 Yelp 没有结果，再用 Google Places 兜底，只接受类型为 restaurant/food 等的门店；  
           - 你从候选列表中选择正确的那一家。  
        2. 若 Yelp 和 Google 均未识别为餐饮门店，则会提示“该地址不是餐厅地址”。  
        3. 选择好餐厅后，在第二步填入当前日均外卖单量与客单价，点击「开始诊断」，生成完整诊断报告。  
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

import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import quote_plus
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
    """
    优先从 st.secrets 读取（Streamlit Cloud / 本地 secrets.toml），
    读取不到则从环境变量中拿。
    """
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
    """
    使用 Google Geocoding API 将地址转为坐标。
    """
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
    """
    使用 Places Find Place API 找到 place_id。
    """
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    params = {
        "key": GOOGLE_API_KEY,
        "input": address,
        "inputtype": "textquery",
        "fields": "place_id,name,geometry"
    }
    r = requests.get(url, params=params, timeout=15)
    data = r.json()
    candidates = data.get("candidates", [])
    if not candidates:
        return None
    return candidates[0]


def google_place_details(place_id: str):
    """
    获取 Place 详情信息（目前主要用于评分、评论数量）。
    Google API 不直接给结构化菜单，这里只取基本信息。
    """
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "name,rating,user_ratings_total,formatted_address"
    }
    r = requests.get(url, params=params, timeout=15)
    return r.json().get("result", {})


def fetch_google_dinein_menu(address: str) -> pd.DataFrame:
    """
    占位函数：
    Google 官方 API 目前不直接提供结构化菜单。
    如果后续你想解析 Google Maps 网页的菜单，可以在这里扩展 HTML 解析逻辑。

    当前先返回空 DataFrame，后面评分逻辑会自动给中性评分。
    """
    return pd.DataFrame(columns=["name", "price", "category", "channel"])


# ===================== Yelp API 相关 ===================== #

def fetch_yelp_business_by_location(address: str):
    """
    通过地址 → 坐标 → Yelp 搜索附近评分最高的一家店视为目标店。
    """
    if not YELP_API_KEY:
        return None

    lat, lng = google_geocode(address)
    if lat is None or lng is None:
        return None

    headers = {"Authorization": f"Bearer {YELP_API_KEY}"}
    url = "https://api.yelp.com/v3/businesses/search"
    params = {
        "latitude": lat,
        "longitude": lng,
        "limit": 1,
        "sort_by": "best_match"
    }
    r = requests.get(url, headers=headers, params=params, timeout=15)
    data = r.json()
    businesses = data.get("businesses", [])
    if not businesses:
        return None

    biz = businesses[0]
    return {
        "id": biz["id"],
        "name": biz["name"],
        "rating": biz.get("rating", None),
        "review_count": biz.get("review_count", 0),
        "price_level": biz.get("price", ""),
        "categories": [c["title"] for c in biz.get("categories", [])],
        "lat": biz["coordinates"]["latitude"],
        "lng": biz["coordinates"]["longitude"],
        "address": ", ".join(biz["location"].get("display_address", []))
    }


def fetch_yelp_competitors(lat: float, lng: float, term: str = "", radius_m: int = 1000) -> pd.DataFrame:
    """
    使用 Yelp 搜索 1km 内竞对。
    """
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
    # 转 km
    if "distance_m" in df.columns:
        df["distance_km"] = df["distance_m"] / 1000.0
    return df


# ===================== 外卖平台网页搜索 & 菜单解析 ===================== #

def search_duckduckgo(query: str, max_results: int = 5):
    """
    使用 DuckDuckGo 的 HTML 结果页面做简单搜索。
    这是公开 Web 搜索，不依赖任何私有 API。
    """
    url = "https://duckduckgo.com/html/"
    params = {"q": query}
    r = requests.get(url, params=params, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
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
    """
    通过搜索找到 Doordash / UberEats 的店铺链接（尽力而为）。
    """
    dd_link = None
    ue_link = None

    # 从地址里抽一点简单的 city / extra
    query_base = f'"{restaurant_name}" {address}'

    # 搜索 Doordash
    dd_results = search_duckduckgo(query_base + " site:doordash.com")
    for link in dd_results:
        if "doordash.com" in link:
            dd_link = link
            break

    # 搜索 UberEats
    ue_results = search_duckduckgo(query_base + " site:ubereats.com")
    for link in ue_results:
        if "ubereats.com" in link:
            ue_link = link
            break

    return dd_link, ue_link


def parse_doordash_menu(url: str) -> pd.DataFrame:
    """
    非官方 Doordash 菜单解析（只读公开 HTML，尽量从中提取品名和价格）。
    Doordash 页面结构经常变动，此处只是一个“能用就赚到”的尝试。
    解析失败时返回空 DataFrame。
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = requests.get(url, headers=headers, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")

        items = []
        # 这里的 CSS 选择器只是示例，未来可能需要根据实际页面调整
        for block in soup.find_all(["div", "article"]):
            name_tag = block.find("h3")
            if not name_tag:
                continue
            name = name_tag.get_text(strip=True)
            # 找价格
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

        return pd.DataFrame(items) if items else pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])
    except Exception:
        return pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])


def parse_ubereats_menu(url: str) -> pd.DataFrame:
    """
    非官方 UberEats 菜单解析（同样只读公开 HTML）。
    结构也可能变动，失败时返回空表。
    """
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

        return pd.DataFrame(items) if items else pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])
    except Exception:
        return pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])


# ===================== 分析逻辑（五大维度） ===================== #

def compute_menu_structure_score(df_all: pd.DataFrame):
    """
    维度1：菜单结构健康度（0–100）
    简单规则：菜太少/太多、类目过多、缺套餐 → 扣分。
    """
    tips = []

    if df_all is None or df_all.empty:
        return 55.0, ["未成功获取外卖菜单数据，暂用中性偏保守评分。"]

    total_items = len(df_all)
    num_categories = df_all["category"].nunique() if "category" in df_all.columns else 1

    score = 100.0

    if total_items < 10:
        score -= 15
        tips.append("外卖菜单单品过少，用户选择有限，建议补充 2–3 个高毛利 Star Item。")
    elif total_items > 60:
        score -= 25
        tips.append("外卖菜单单品超过 60 个，容易导致选择困难，建议精简和合并部分菜品。")

    if num_categories > 8:
        score -= 15
        tips.append("菜单类别过多，建议压缩到 5–7 个主类目，突出主力品类。")

    has_combo = False
    for c in df_all.get("category", pd.Series()).dropna().astype(str):
        if "combo" in c.lower() or "套餐" in c:
            has_combo = True
            break
    if not has_combo:
        score -= 10
        tips.append("缺少套餐/组合菜单，建议设计 2–3 个客单价更高的套餐组合，提升客单价。")

    return max(score, 0), tips


def compute_pricing_score(df_dinein: pd.DataFrame, df_delivery: pd.DataFrame):
    """
    维度2：定价与客单价策略（0–100）
    堂食 vs 外卖的加价率。
    没有堂食数据时使用中性评分。
    """
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
        tips.append(f"外卖整体加价率约 {avg_markup:.0%}，偏低，建议适当提高到 15% 左右以覆盖平台与配送成本。")
    elif avg_markup > 0.35:
        score -= 20
        tips.append(f"外卖整体加价率约 {avg_markup:.0%}，偏高，可能影响转化率，建议控制在 15%–30% 区间。")
    else:
        tips.append(f"外卖加价率约 {avg_markup:.0%}，整体合理。")

    return max(score, 0), tips


def compute_promotion_score(has_dd_link: bool, has_ue_link: bool):
    """
    维度3：活动体系（0–100）
    当前没有深入解析活动，仅用简单逻辑：
    - 上了两个平台 → 评分稍高
    - 一个平台 → 中性
    - 没上 → 偏低
    未来可扩展解析 BOGO / 满减 等信息。
    """
    tips = []
    if not has_dd_link and not has_ue_link:
        score = 45.0
        tips.append("暂未发现 Doordash / UberEats 店铺链接，外卖渠道基础需要先补齐。")
    elif has_dd_link and has_ue_link:
        score = 70.0
        tips.append("已覆盖主流外卖平台，后续可重点设计分平台差异化优惠与老客复购活动。")
    else:
        score = 60.0
        tips.append("外卖平台部分覆盖，建议同步拓展至主流平台，并制定一致的价格与活动策略。")

    tips.append("当前版本未读取具体活动内容，建议上线后搭配：首单减免、午晚高峰满减、老客券包等组合玩法。")
    return score, tips


def compute_competitor_score(df_comp: pd.DataFrame, restaurant_rating: float):
    """
    维度4：竞对压力指数（0–100）
    看自己评分 vs 周边均值。
    """
    tips = []

    if df_comp is None or df_comp.empty or restaurant_rating is None:
        return 60.0, ["竞对或本店评分数据不完整，暂用中性评分。"]

    avg_comp_rating = df_comp["rating"].mean()
    diff = restaurant_rating - avg_comp_rating
    score = 60.0 + diff * 10
    score = max(min(score, 100.0), 0.0)

    if diff >= 0.2:
        tips.append(f"本店 Yelp 评分 {restaurant_rating:.1f} 高于附近竞对均值 {avg_comp_rating:.1f}，口碑具备优势，可以在外卖详情页更突出。")
    elif diff <= -0.2:
        tips.append(f"本店 Yelp 评分 {restaurant_rating:.1f} 低于附近竞对均值 {avg_comp_rating:.1f}，建议通过服务体验、包装、好评激励活动快速拉升评分。")
    else:
        tips.append("本店评分与附近竞对大致持平，建议通过菜品照片、文案与活动玩法做差异化。")

    return score, tips


def compute_coverage_score():
    """
    维度5：配送覆盖 & 圈层（0–100）
    当前版本未接入真实配送半径，给一个中性偏乐观评分。
    未来可以根据平台 API 或自建数据做更精细的评估。
    """
    score = 70.0
    tips = [
        "从地理位置和商圈结构的通用经验看，配送覆盖具备一定潜力，"
        "后续可结合实际平台配送半径与学校/写字楼密度做进一步量化。"
    ]
    return score, tips


def compute_growth_rate(menu_score, price_score, promo_score, comp_score, coverage_score) -> float:
    """
    汇总五大维度，计算“潜在增长率”（0~1），并限制在 MIN_GROWTH ~ MAX_GROWTH 区间。
    """
    weighted = (
        0.20 * menu_score +
        0.15 * price_score +
        0.25 * promo_score +
        0.15 * comp_score +
        0.25 * coverage_score
    ) / 100.0
    growth_rate = MIN_GROWTH + (MAX_GROWTH - MIN_GROWTH) * weighted
    return growth_rate


# ===================== 核心分析管线 ===================== #

def analyze_restaurant(address: str, avg_orders: float, avg_ticket: float):
    """
    核心流程：
    1. Yelp 找到目标店 & 竞对
    2. Google 获取 Place 信息
    3. 搜索 Doordash / UberEats 链接并尝试解析菜单
    4. 计算五大维度评分
    5. 预估外卖营业额提升空间
    """
    # 1. Yelp 基础信息
    yelp_info = fetch_yelp_business_by_location(address)
    if not yelp_info:
        raise RuntimeError("根据地址未在 Yelp 找到匹配餐厅，请检查地址是否正确。")

    # 2. Google Place 信息
    place_info = None
    place = google_find_place(address)
    if place and place.get("place_id"):
        place_info = google_place_details(place["place_id"])

    # 3. 竞对（按 Yelp）
    comp_df = fetch_yelp_competitors(yelp_info["lat"], yelp_info["lng"])

    # 4. 菜单数据
    dinein_df = fetch_google_dinein_menu(address)  # 当前为空占位
    # 搜索外卖平台链接
    dd_link, ue_link = find_delivery_links(yelp_info["name"], yelp_info["address"])
    dd_df = parse_doordash_menu(dd_link) if dd_link else pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])
    ue_df = parse_ubereats_menu(ue_link) if ue_link else pd.DataFrame(columns=["name", "price", "category", "channel", "tags"])

    all_df = pd.concat([dinein_df, dd_df, ue_df], ignore_index=True) if not (dd_df.empty and ue_df.empty) else pd.DataFrame()

    # 5. 评分与建议
    menu_score, menu_tips = compute_menu_structure_score(all_df)
    price_score, price_tips = compute_pricing_score(dinein_df, dd_df if not dd_df.empty else ue_df)
    promo_score, promo_tips = compute_promotion_score(has_dd_link=dd_link is not None, has_ue_link=ue_link is not None)
    comp_score, comp_tips = compute_competitor_score(comp_df, yelp_info.get("rating", None))
    coverage_score, coverage_tips = compute_coverage_score()

    growth_rate = compute_growth_rate(menu_score, price_score, promo_score, comp_score, coverage_score)

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
        },
        "tips": {
            "菜单结构": menu_tips,
            "定价与客单价": price_tips,
            "活动体系": promo_tips,
            "竞对压力": comp_tips,
            "覆盖与圈层": coverage_tips,
        },
        "growth_rate": growth_rate,
        "current_daily_revenue": current_daily_revenue,
        "potential_daily_revenue": potential_daily_revenue,
        "revenue_uplift_daily": revenue_uplift_daily,
        "revenue_uplift_monthly": revenue_uplift_monthly,
    }

    return result


# ===================== Streamlit UI ===================== #

with st.form("input_form"):
    st.subheader("📍 请输入餐厅基础数据")

    address = st.text_input("餐厅地址（用于匹配 Yelp / Google / 外卖平台）", "")
    col1, col2 = st.columns(2)
    with col1:
        avg_orders = st.number_input("当前日均外卖单量（单）", min_value=0.0, value=30.0, step=1.0)
    with col2:
        avg_ticket = st.number_input("当前外卖客单价（美元）", min_value=0.0, value=25.0, step=1.0)

    submitted = st.form_submit_button("🚀 开始诊断")

if submitted:
    if not address.strip():
        st.error("请输入餐厅地址。")
    else:
        try:
            with st.spinner("正在基于 Yelp / Google / 外卖平台数据进行诊断..."):
                result = analyze_restaurant(address, avg_orders, avg_ticket)

            # 顶部 KPI 概览
            st.subheader("📊 诊断结果总览")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric(
                    "当前日外卖营业额（估算）",
                    f"${result['current_daily_revenue']:.0f}"
                )
            with col_b:
                st.metric(
                    "优化后日外卖营业额（预测）",
                    f"${result['potential_daily_revenue']:.0f}"
                )
            with col_c:
                st.metric(
                    "月度可提升外卖营业额（预测）",
                    f"+${result['revenue_uplift_monthly']:.0f}"
                )

            st.write(
                f"综合菜单结构、定价策略、活动体系、竞对压力与配送覆盖情况，"
                f"系统预估通过精细化运营，可带来约 **{result['growth_rate']*100:.1f}%** 的外卖营业额增长空间。"
            )

            # 五大维度评分
            st.subheader("🧬 五大维度诊断评分")
            score_df = pd.DataFrame(
                {
                    "维度": list(result["scores"].keys()),
                    "得分": list(result["scores"].values()),
                }
            )
            st.bar_chart(score_df.set_index("维度"))

            # 维度建议
            st.subheader("🩺 分维度运营建议")
            for dim, tips in result["tips"].items():
                with st.expander(f"{dim} · 诊断建议"):
                    for t in tips:
                        st.markdown(f"- {t}")

            # Yelp / Google 基本信息
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

            # 菜单表
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

            # 竞对
            st.subheader("🏁 附近竞对概览（来自 Yelp）")
            if result["competitors"].empty:
                st.write("未获取到竞对数据。")
            else:
                st.dataframe(result["competitors"])

            st.info(
                "当前版本已接入真实 Yelp / Google API，外卖平台菜单解析基于公开网页结构，"
                "若平台改版或个别页面结构特殊，可能出现解析不到菜单的情况，评分会自动回退为中性。"
            )

        except Exception as e:
            st.error(f"诊断过程中出现错误：{e}")

else:
    st.markdown(
        """
        ### 使用说明
        1. 填写餐厅地址 + 当前日均外卖单量 + 外卖客单价  
        2. 系统会通过 **Yelp / Google API** 定位店铺与竞对，通过公开网页搜索尝试找到 Doordash / UberEats 店铺页面；  
        3. 在能解析到的前提下，对菜单结构、价格策略、外卖平台覆盖、竞对情况做量化评分；  
        4. 输出一份「外卖营业额可提升空间」的预测结果 + 分维度运营建议，可直接用于和老板/客户沟通。  
        """
    )

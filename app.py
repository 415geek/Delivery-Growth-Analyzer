import streamlit as st
import pandas as pd
import requests
import googlemaps
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
import math
import time

# ---------------------------
# 基础配置
# ---------------------------
st.set_page_config(
    page_title="Restaurant Local SEO & Competitor Analyzer",
    layout="wide",
)

st.title("🍜 Restaurant Local SEO & Competitor Analyzer")
st.write(
    "复制 Owner.com 风格的本地餐厅在线健康检查：\n"
    "- 自动找竞争对手\n"
    "- 评估 Google 商家资料完整度\n"
    "- 检查网站内容/SEO 基础\n"
    "- 模拟本地搜索排名\n"
    "- 粗算潜在营收损失"
)

# ---------------------------
# 侧边栏：API Key & 参数
# ---------------------------
# ---------------------------
# 从 secrets 读取 API Key
# ---------------------------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
YELP_API_KEY = st.secrets.get("YELP_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")  # 以后要用可以直接拿

if not GOOGLE_API_KEY:
    st.error("缺少 GOOGLE_API_KEY，请在 Streamlit Secrets 中配置。")
    st.stop()

default_radius_km = st.sidebar.slider(
    "竞争对手搜索半径（公里）", 0.5, 10.0, 3.0, 0.5
)

avg_order_value = st.sidebar.number_input(
    "平均客单价（USD）", min_value=5.0, max_value=200.0, value=40.0, step=1.0
)
assumed_ctr = st.sidebar.slider(
    "点击率假设（用户看到你的结果后点进来的比例）",
    0.05, 0.5, 0.15, 0.01
)
assumed_conv = st.sidebar.slider(
    "下单转化率假设（点进网站/资料后下单的比例）",
    0.05, 0.5, 0.2, 0.01
)

st.sidebar.markdown("---")



# ---------------------------
# 工具函数（缓存）
# ---------------------------

@st.cache_data(show_spinner=False)
def gm_client(key: str):
    return googlemaps.Client(key=key)


@st.cache_data(show_spinner=False)
def google_places_search(api_key: str, query: str) -> List[Dict[str, Any]]:
    gmaps = gm_client(api_key)
    result = gmaps.places(query=query)
    return result.get("results", [])


@st.cache_data(show_spinner=False)
def google_place_details(api_key: str, place_id: str) -> Dict[str, Any]:
    gmaps = gm_client(api_key)
    fields = [
        "name",
        "formatted_address",
        "formatted_phone_number",
        "geometry",
        "rating",
        "user_ratings_total",
        "types",
        "opening_hours",
        "website",
        "price_level",
        "photos",
    ]
    result = gmaps.place(place_id=place_id, fields=fields)
    return result.get("result", {})


@st.cache_data(show_spinner=False)
def google_places_nearby(
    api_key: str, lat: float, lng: float, radius_m: int, type_: str = "restaurant"
) -> List[Dict[str, Any]]:
    gmaps = gm_client(api_key)
    result = gmaps.places_nearby(
        location=(lat, lng), radius=radius_m, type=type_
    )
    return result.get("results", [])


@st.cache_data(show_spinner=False)
def fetch_html(url: str) -> Optional[str]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Restaurant-Analyzer)"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.text
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def serpapi_google_maps_search(
    serpapi_key: str, query: str, lat: float, lng: float, zoom: float = 13.0
) -> Dict[str, Any]:
    """
    使用 SerpAPI 的 Google Maps 引擎做真实本地搜索。
    需要付费/限额，用户自行申请 key。
    """
    url = "https://serpapi.com/search"
    ll_param = f"@{lat},{lng},{zoom}z"
    params = {
        "engine": "google_maps",
        "type": "search",
        "q": query,
        "ll": ll_param,
        "api_key": serpapi_key,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------
# 评分函数
# ---------------------------

def score_gbp_profile(place: Dict[str, Any]) -> Dict[str, Any]:
    """
    简化版 Google 商家资料评分，总分 40 分。
    你可以根据自己策略继续细化。
    """
    score = 0
    checks = {}

    # 1. 名称 & 地址
    has_name = bool(place.get("name"))
    has_address = bool(place.get("formatted_address"))
    pts = 4 if (has_name and has_address) else 0
    score += pts
    checks["名称/地址完整"] = (pts, has_name and has_address)

    # 2. 电话
    has_phone = bool(place.get("formatted_phone_number"))
    pts = 4 if has_phone else 0
    score += pts
    checks["电话"] = (pts, has_phone)

    # 3. 营业时间
    opening_hours = place.get("opening_hours", {})
    has_hours = bool(opening_hours.get("weekday_text")) or opening_hours.get(
        "open_now"
    ) is not None
    pts = 4 if has_hours else 0
    score += pts
    checks["营业时间"] = (pts, has_hours)

    # 4. 网站
    has_website = bool(place.get("website"))
    pts = 4 if has_website else 0
    score += pts
    checks["网站链接"] = (pts, has_website)

    # 5. 评分 & 评论数
    rating = place.get("rating")
    reviews = place.get("user_ratings_total", 0)
    has_reviews = rating is not None and reviews >= 10
    pts = 6 if has_reviews else 0
    score += pts
    checks["评分 & ≥10条评论"] = (pts, has_reviews)

    # 6. 类别
    types_ = place.get("types", [])
    has_category = any(t for t in types_ if t != "point_of_interest")
    pts = 6 if has_category else 0
    score += pts
    checks["类别设置"] = (pts, has_category)

    # 7. 价格等级
    has_price_level = place.get("price_level") is not None
    pts = 4 if has_price_level else 0
    score += pts
    checks["价格区间"] = (pts, has_price_level)

    # 8. 照片
    photos = place.get("photos", [])
    has_photos = len(photos) > 0
    pts = 8 if has_photos else 0
    score += pts
    checks["照片/图片"] = (pts, has_photos)

    return {"score": score, "checks": checks}


def score_website_basic(url: str, html: Optional[str]) -> Dict[str, Any]:
    """
    简化版网站评分，总分 40 分。
    主要看 SEO 基础 + 文本量 + 是否有关键词等。
    """
    if not url or not html:
        return {
            "score": 0,
            "checks": {"无法访问网站": (0, False)},
        }

    soup = BeautifulSoup(html, "lxml")
    score = 0
    checks = {}

    # 1. Title
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    has_title = bool(title)
    pts = 6 if has_title else 0
    score += pts
    checks["有页面标题（title）"] = (pts, has_title)

    # 2. Meta Description
    desc_tag = soup.find("meta", attrs={"name": "description"})
    has_desc = bool(desc_tag and desc_tag.get("content"))
    pts = 6 if has_desc else 0
    score += pts
    checks["有 Meta Description"] = (pts, has_desc)

    # 3. H1
    h1 = soup.find("h1")
    has_h1 = bool(h1 and h1.get_text(strip=True))
    pts = 4 if has_h1 else 0
    score += pts
    checks["有 H1 标题"] = (pts, has_h1)

    # 4. 文本总量
    texts = soup.get_text(separator=" ", strip=True)
    word_count = len(texts.split())
    has_sufficient_text = word_count >= 300
    pts = 8 if has_sufficient_text else 0
    score += pts
    checks["文本量 ≥ 300 词"] = (pts, has_sufficient_text)

    # 5. 联系方式
    has_phone_text = any(x in texts for x in ["(", ")", "-", "+1"])
    pts = 4 if has_phone_text else 0
    score += pts
    checks["页面上能看到电话"] = (pts, has_phone_text)

    # 6. 菜品/餐厅关键词（简单匹配）
    keywords = ["chinese", "asian", "noodle", "rice", "dumpling", "chicken"]
    kw_hit = any(kw.lower() in texts.lower() for kw in keywords)
    pts = 6 if kw_hit else 0
    score += pts
    checks["文本包含菜品/菜系关键词"] = (pts, kw_hit)

    # 7. https
    parsed = urlparse(url)
    has_https = parsed.scheme == "https"
    pts = 6 if has_https else 0
    score += pts
    checks["使用 HTTPS"] = (pts, has_https)

    return {"score": score, "checks": checks, "word_count": word_count, "title": title}


def estimate_revenue_loss(
    monthly_search_volume: int,
    rank_bucket: str,
    avg_order_value: float,
    ctr: float,
    conv: float,
) -> float:
    """
    非精确模型，只是给老板一个大概感觉。
    简单机制：
    - Top 3：可以吃到 100% 潜在流量
    - 4-10：只能吃到 40%
    - 未上榜：10%
    用这个对比“理想状态 vs 当前状态”的差额。
    """
    ideal_customers = monthly_search_volume * ctr * conv
    if rank_bucket == "top3":
        current_factor = 1.0
    elif rank_bucket == "4-10":
        current_factor = 0.4
    else:
        current_factor = 0.1

    current_customers = ideal_customers * current_factor
    potential_extra_customers = ideal_customers - current_customers
    monthly_loss = potential_extra_customers * avg_order_value
    return monthly_loss


def infer_rank_from_serpapi(
    serp_json: Dict[str, Any], business_name: str
) -> Optional[int]:
    """
    从 SerpAPI Google Maps 结果中找到当前餐厅的名次。
    """
    results = serp_json.get("local_results") or serp_json.get("places_results") or []
    for idx, res in enumerate(results, start=1):
        name = res.get("title") or res.get("name", "")
        if name and business_name.lower() in name.lower():
            return idx
    return None


# ---------------------------
# 主交互：输入餐厅信息
# ---------------------------

st.markdown("## 1️⃣ 输入餐厅信息")

col1, col2 = st.columns(2)
with col1:
    restaurant_name = st.text_input("餐厅名称（Restaurant Name）", "")
with col2:
    city_region = st.text_input("城市/区域（例如：Outer Sunset, San Francisco）", "")

website_url = st.text_input(
    "餐厅官网 URL（如 https://example.com）",
    "",
)

keywords_input = st.text_input(
    "核心关键词（逗号分隔，例如：best chinese food outer sunset, best asian food west portal）",
    "best chinese food outer sunset, best asian food outer sunset",
)

monthly_search_volume = st.number_input(
    "估算每个核心关键词的月搜索量（粗略统一值）",
    min_value=50,
    max_value=50000,
    value=500,
    step=50,
    help="更精确可以以后接 Google Ads Keyword Planner 或第三方关键词 API。",
)

run_btn = st.button("🚀 运行分析")

if run_btn:
    if not google_api_key:
        st.error("请先在左侧输入 Google API Key。")
        st.stop()

    if not restaurant_name or not city_region:
        st.error("请填写餐厅名称和城市/区域。")
        st.stop()

    query = f"{restaurant_name} {city_region}"
    with st.spinner(f"在 Google Places 中搜索：{query}"):
        places = google_places_search(google_api_key, query)

    if not places:
        st.error("Google Places 未找到匹配餐厅，请检查名称和城市。")
        st.stop()

    # 先用搜索结果中第一个
    target = places[0]
    place_id = target["place_id"]

    with st.spinner("获取餐厅详情（Google Place Details）..."):
        place_detail = google_place_details(google_api_key, place_id)

    st.success(f"已找到餐厅：**{place_detail.get('name', 'Unknown')}**")

    # ---------------------------
    # 显示基础信息
    # ---------------------------
    st.markdown("### 🧾 基本信息（来自 Google Places）")
    info_cols = st.columns(3)
    with info_cols[0]:
        st.write("**名称**:", place_detail.get("name"))
        st.write("**地址**:", place_detail.get("formatted_address"))
    with info_cols[1]:
        st.write("**电话**:", place_detail.get("formatted_phone_number", "N/A"))
        st.write("**评分**:", place_detail.get("rating", "N/A"))
        st.write("**评论数**:", place_detail.get("user_ratings_total", "N/A"))
    with info_cols[2]:
        st.write("**价格等级**:", place_detail.get("price_level", "N/A"))
        st.write("**官网（来自 Google）**:", place_detail.get("website", "N/A"))

    geometry = place_detail.get("geometry", {}).get("location", {})
    lat = geometry.get("lat")
    lng = geometry.get("lng")

    # ---------------------------
    # 竞争对手搜索
    # ---------------------------
    st.markdown("## 2️⃣ 附近竞争对手（Google Places Nearby）")

    if lat is None or lng is None:
        st.warning("未能从 Google 获取经纬度，无法搜索附近竞争对手。")
        competitors = []
    else:
        radius_m = int(default_radius_km * 1000)
        with st.spinner("搜索附近餐厅作为竞争对手..."):
            competitors = google_places_nearby(
                google_api_key, lat, lng, radius_m, type_="restaurant"
            )

    if competitors:
        comp_data = []
        for c in competitors:
            comp_data.append(
                {
                    "Name": c.get("name"),
                    "Address": c.get("vicinity"),
                    "Rating": c.get("rating", None),
                    "Reviews": c.get("user_ratings_total", 0),
                }
            )
        df_comp = pd.DataFrame(comp_data)
        # 排除自己
        df_comp = df_comp[
            df_comp["Name"].str.lower()
            != place_detail.get("name", "").lower()
        ]
        df_comp_sorted = df_comp.sort_values(
            by=["Rating", "Reviews"], ascending=[False, False]
        ).reset_index(drop=True)
        st.dataframe(df_comp_sorted, use_container_width=True)
    else:
        st.info("未找到竞争对手（可能半径太小或 API 限制）。")

    # ---------------------------
    # Google 商家资料评分
    # ---------------------------
    st.markdown("## 3️⃣ Google 商家资料评分（40 分制）")

    gbp_result = score_gbp_profile(place_detail)
    st.metric("Google 商家资料得分", f"{gbp_result['score']} / 40")

    gbp_rows = []
    for label, (pts, ok) in gbp_result["checks"].items():
        gbp_rows.append(
            {
                "检查项": label,
                "得分": pts,
                "状态": "✅ 完成" if ok else "❌ 缺失/不完整",
            }
        )
    df_gbp = pd.DataFrame(gbp_rows)
    st.table(df_gbp)

    # ---------------------------
    # 网站评分
    # ---------------------------
    st.markdown("## 4️⃣ 网站内容 & 体验评分（40 分制）")

    # 优先用 Google 返回的网站，如果用户没填或不一致，可以自己改
    effective_website = website_url or place_detail.get("website")

    if not effective_website:
        st.warning("未提供网站 URL，也无法从 Google 获取，网站评分为 0。")
        web_result = {"score": 0, "checks": {"无网站": (0, False)}}
    else:
        with st.spinner(f"抓取网站：{effective_website}"):
            html = fetch_html(effective_website)

        web_result = score_website_basic(effective_website, html)

    st.metric("网站得分", f"{web_result['score']} / 40")
    web_rows = []
    for label, (pts, ok) in web_result["checks"].items():
        web_rows.append(
            {
                "检查项": label,
                "得分": pts,
                "状态": "✅ 是" if ok else "❌ 否",
            }
        )
    df_web = pd.DataFrame(web_rows)
    st.table(df_web)

    # ---------------------------
    # 本地关键词排名模拟 + 营收损失估算
    # ---------------------------
    st.markdown("## 5️⃣ 本地关键词排名 & 潜在营收损失")

    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    rank_results = []

    if not keywords:
        st.info("未提供关键词，跳过排名模拟和营收估算。")
    else:
        for kw in keywords:
            st.write(f"### 关键词：**{kw}**")
            # rank_bucket: top3 / 4-10 / none
            rank_bucket = "none"
            rank_position = None

            if serpapi_key and lat is not None and lng is not None:
                with st.spinner(f"使用 SerpAPI 查询 Google Maps 排名：{kw}"):
                    try:
                        serp_json = serpapi_google_maps_search(
                            serpapi_key, kw, lat, lng
                        )
                        rank_position = infer_rank_from_serpapi(
                            serp_json, place_detail.get("name", "")
                        )
                        if rank_position is not None:
                            if rank_position <= 3:
                                rank_bucket = "top3"
                            elif rank_position <= 10:
                                rank_bucket = "4-10"
                            else:
                                rank_bucket = "none"
                    except Exception as e:
                        st.warning(f"SerpAPI 查询出错：{e}")
                        rank_bucket = "none"
                        rank_position = None
            else:
                # 没有 SerpAPI，就提供一个非常粗糙的“模拟”：
                # - 如果你的评分和评论数在附近竞争对手中属于前列，就假设进入 4-10 或 top3。
                if competitors:
                    all_places = competitors + [place_detail]
                    all_places_data = []
                    for p in all_places:
                        all_places_data.append(
                            {
                                "name": p.get("name", ""),
                                "rating": p.get("rating", 0),
                                "reviews": p.get("user_ratings_total", 0),
                            }
                        )
                    df_all = pd.DataFrame(all_places_data)
                    df_all["score"] = (
                        df_all["rating"].fillna(0) * 10
                        + df_all["reviews"].fillna(0) / 10
                    )
                    df_all = df_all.sort_values(
                        by="score", ascending=False
                    ).reset_index(drop=True)
                    positions = (
                        df_all["name"]
                        .str.lower()
                        .tolist()
                    )
                    if place_detail.get("name", "").lower() in positions:
                        pos = positions.index(
                            place_detail.get("name", "").lower()
                        ) + 1
                        rank_position = pos
                        if pos <= 3:
                            rank_bucket = "top3"
                        elif pos <= 10:
                            rank_bucket = "4-10"
                        else:
                            rank_bucket = "none"
                    else:
                        rank_bucket = "none"

            # 计算营收损失
            monthly_loss = estimate_revenue_loss(
                monthly_search_volume,
                rank_bucket,
                avg_order_value,
                assumed_ctr,
                assumed_conv,
            )

            st.write(
                f"- 估计当前排名："
                f"{'Top 3' if rank_bucket=='top3' else ('第 4–10 名' if rank_bucket=='4-10' else '未进入前 10')}"
                f"{'' if rank_position is None else f'（推测名次：{rank_position}）'}"
            )
            st.write(
                f"- 粗略估计：由于没有在理想位置，**每月可能少赚约 ${monthly_loss:,.0f}**"
            )

            rank_results.append(
                {
                    "关键词": kw,
                    "预估名次": rank_position,
                    "名次区间": rank_bucket,
                    "预估月损失($)": round(monthly_loss, 2),
                }
            )

        if rank_results:
            st.markdown("#### 关键词 & 营收损失汇总")
            df_rank = pd.DataFrame(rank_results)
            st.dataframe(df_rank, use_container_width=True)

    # ---------------------------
    # 总结
    # ---------------------------
    st.markdown("## 6️⃣ 总体在线健康总结")

    total_score = gbp_result["score"] + web_result["score"]
    st.metric("综合得分（Profile + Website）", f"{total_score} / 80")

    st.write(
        "- **40 分以下**：在线基础非常弱，基本属于 “Poor” 状态。\n"
        "- **40–60 分**：中等，能被找到，但没有优势。\n"
        "- **60 分以上**：相对健康，但仍有优化空间，特别是关键词布局和活动推广。\n"
    )

    st.info(
        "建议下一步：\n"
        "1. 把上面的表格导出给老板（或你自己的客户），逐项勾选优化。\n"
        "2. 后续可以接入：Google Business Profile API 菜单、Yelp API、DoorDash/UberEats 菜单抓取、"
        "以及关键词真实搜索量 API，做成更接近 Owner.com 的完整系统。"
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

import json
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd
import requests
import googlemaps
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from openai import OpenAI

# =========================
# 基本配置 & Secrets
# =========================
st.set_page_config(
    page_title="Restaurant Competitor Analyzer",
    layout="wide",
)

# 从 Streamlit Secrets 读取 API 密钥
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
YELP_API_KEY = st.secrets.get("YELP_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not GOOGLE_API_KEY:
    st.error("缺少 GOOGLE_API_KEY，请先在 Streamlit Secrets 中配置后再刷新。")
    st.stop()

client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# 页面标题
# =========================
st.title("🍜 Restaurant Competitor Analyzer")
st.write(
    "面向餐厅老板的一键体检：\n"
    "- 只需输入地址，自动匹配你的餐厅\n"
    "- 自动找附近竞争对手\n"
    "- 估算堂食/外卖的潜在流失营收\n"
    "- 使用 ChatGPT 做菜系分析运营建议"
)

# =========================
# Session State 初始化
# =========================
if "candidate_places" not in st.session_state:
    st.session_state["candidate_places"] = []
if "selected_index" not in st.session_state:
    st.session_state["selected_index"] = 0
if "run_analysis" not in st.session_state:
    st.session_state["run_analysis"] = False

# =========================
# 工具函数（带缓存）
# =========================

@st.cache_data(show_spinner=False)
def gm_client(key: str):
    return googlemaps.Client(key=key)

@st.cache_data(show_spinner=False)
def google_geocode(api_key: str, address: str) -> List[Dict[str, Any]]:
    gmaps = gm_client(api_key)
    return gmaps.geocode(address)

@st.cache_data(show_spinner=False)
def google_place_details(api_key: str, place_id: str) -> Dict[str, Any]:
    """
    Google Place Details：
    先尝试带 fields，如果 SDK/版本不支持就 fallback 到不带 fields 的调用，
    避免 ValueError.
    """
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
    try:
        result = gmaps.place(place_id=place_id, fields=fields)
        data = result.get("result", result)
    except Exception:
        # 回退：不传 fields，拿全部字段
        result = gmaps.place(place_id=place_id)
        data = result.get("result", result)
    return data

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

# =========================
# 评分函数
# =========================

def score_gbp_profile(place: Dict[str, Any]) -> Dict[str, Any]:
    """简化版 Google 商家资料评分，总分 40 分。"""
    score = 0
    checks = {}

    has_name = bool(place.get("name"))
    has_address = bool(place.get("formatted_address"))
    pts = 4 if (has_name and has_address) else 0
    score += pts
    checks["名称/地址完整"] = (pts, has_name and has_address)

    has_phone = bool(place.get("formatted_phone_number"))
    pts = 4 if has_phone else 0
    score += pts
    checks["电话"] = (pts, has_phone)

    opening_hours = place.get("opening_hours", {})
    has_hours = bool(opening_hours.get("weekday_text")) or opening_hours.get(
        "open_now"
    ) is not None
    pts = 4 if has_hours else 0
    score += pts
    checks["营业时间"] = (pts, has_hours)

    has_website = bool(place.get("website"))
    pts = 4 if has_website else 0
    score += pts
    checks["网站链接"] = (pts, has_website)

    rating = place.get("rating")
    reviews = place.get("user_ratings_total", 0)
    has_reviews = rating is not None and reviews >= 10
    pts = 6 if has_reviews else 0
    score += pts
    checks["评分 & ≥10条评论"] = (pts, has_reviews)

    types_ = place.get("types", [])
    has_category = any(t for t in types_ if t != "point_of_interest")
    pts = 6 if has_category else 0
    score += pts
    checks["类别设置"] = (pts, has_category)

    has_price_level = place.get("price_level") is not None
    pts = 4 if has_price_level else 0
    score += pts
    checks["价格区间"] = (pts, has_price_level)

    photos = place.get("photos", [])
    has_photos = len(photos) > 0
    pts = 8 if has_photos else 0
    score += pts
    checks["照片/图片"] = (pts, has_photos)

    return {"score": score, "checks": checks}

def score_website_basic(url: str, html: Optional[str]) -> Dict[str, Any]:
    """简化版网站评分，总分 40 分 + 返回文本摘要。"""
    if not url or not html:
        return {
            "score": 0,
            "checks": {"无法访问网站": (0, False)},
            "word_count": 0,
            "title": "",
            "text_snippet": "",
        }

    soup = BeautifulSoup(html, "lxml")
    score = 0
    checks = {}

    texts = soup.get_text(separator=" ", strip=True)
    word_count = len(texts.split())
    text_snippet = texts[:3000]

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    has_title = bool(title)
    pts = 6 if has_title else 0
    score += pts
    checks["有页面标题（title）"] = (pts, has_title)

    desc_tag = soup.find("meta", attrs={"name": "description"})
    has_desc = bool(desc_tag and desc_tag.get("content"))
    pts = 6 if has_desc else 0
    score += pts
    checks["有 Meta Description"] = (pts, has_desc)

    h1 = soup.find("h1")
    has_h1 = bool(h1 and h1.get_text(strip=True))
    pts = 4 if has_h1 else 0
    score += pts
    checks["有 H1 标题"] = (pts, has_h1)

    has_sufficient_text = word_count >= 300
    pts = 8 if has_sufficient_text else 0
    score += pts
    checks["文本量 ≥ 300 词"] = (pts, has_sufficient_text)

    has_phone_text = any(x in texts for x in ["(", ")", "-", "+1"])
    pts = 4 if has_phone_text else 0
    score += pts
    checks["页面上能看到电话"] = (pts, has_phone_text)

    keywords = [
        "chinese", "cantonese", "szechuan", "sichuan", "shanghai",
        "dim sum", "noodle", "rice", "dumpling", "hot pot", "bbq"
    ]
    kw_hit = any(kw.lower() in texts.lower() for kw in keywords)
    pts = 6 if kw_hit else 0
    score += pts
    checks["文本包含菜品/菜系关键词"] = (pts, kw_hit)

    parsed = urlparse(url)
    has_https = parsed.scheme == "https"
    pts = 6 if has_https else 0
    score += pts
    checks["使用 HTTPS"] = (pts, has_https)

    return {
        "score": score,
        "checks": checks,
        "word_count": word_count,
        "title": title,
        "text_snippet": text_snippet,
    }

def estimate_revenue_loss(
    monthly_search_volume: int,
    rank_bucket: str,
    avg_order_value: float,
    channel: str = "dine-in",
) -> float:
    """粗略营收损失估算（内部 CTR/转化率假设）。"""
    if channel == "delivery":
        ctr = 0.18
        conv = 0.35
    else:
        ctr = 0.12
        conv = 0.25

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
    """从 SerpAPI Google Maps 结果中找到当前餐厅名次。"""
    results = serp_json.get("local_results") or serp_json.get("places_results") or []
    for idx, res in enumerate(results, start=1):
        name = res.get("title") or res.get("name", "")
        if name and business_name.lower() in name.lower():
            return idx
    return None

# =========================
# ChatGPT 深度分析函数
# =========================

def llm_deep_analysis(
    place_detail: Dict[str, Any],
    gbp_result: Dict[str, Any],
    web_result: Dict[str, Any],
    competitors_df: Optional[pd.DataFrame],
    rank_results: List[Dict[str, Any]],
    monthly_search_volume: int,
    dine_in_aov: float,
    delivery_aov: float,
) -> str:
    if client is None:
        return "未配置 OPENAI_API_KEY，无法调用 ChatGPT。"

    comp_json = []
    if competitors_df is not None and not competitors_df.empty:
        sub = competitors_df.head(5)
        comp_json = sub.to_dict(orient="records")

    payload = {
        "restaurant": {
            "name": place_detail.get("name"),
            "address": place_detail.get("formatted_address"),
            "phone": place_detail.get("formatted_phone_number"),
            "types": place_detail.get("types", []),
            "rating": place_detail.get("rating"),
            "reviews": place_detail.get("user_ratings_total"),
            "price_level": place_detail.get("price_level"),
        },
        "gbp_score": gbp_result["score"],
        "gbp_checks": gbp_result["checks"],
        "website_score": web_result["score"],
        "website_title": web_result.get("title", ""),
        "website_word_count": web_result.get("word_count", 0),
        "competitors": comp_json,
        "rank_results": rank_results,
        "assumptions": {
            "monthly_search_volume_per_keyword": monthly_search_volume,
            "dine_in_aov": dine_in_aov,
            "delivery_aov": delivery_aov,
        },
    }

    text_snippet = web_result.get("text_snippet", "")

    system_msg = (
        "你是一名专门服务北美餐馆的本地营销和外卖运营顾问，曾任职于麦肯锡一个专门做餐饮分析的部门，"
        "非常了解世界各地的菜系，尤其在中餐菜系的细分领域属于行业权威，如粤菜、茶餐厅、川菜、湘菜、东北菜、上海菜等，"
        "熟悉 Google 本地搜索和 UberEats/DoorDash/Grubhub/Hungrypanda/Fantuan 等平台的运营逻辑。"
        "请用简体中文回答，但在需要时可加少量英文术语。"
    )

    user_msg = f"""
这是一个餐厅的在线数据，请你做**多维深度分析**并给出细分菜系判断与运营建议。

【结构化数据 JSON】
{json.dumps(payload, ensure_ascii=False, indent=2)}

【网站文本片段（最多 3000 字符）】
{text_snippet}

请完成以下任务（分段输出，方便老板阅读）：

1. **菜系细分判断**
   - 判断该店最有可能属于哪一类：如正宗川菜馆、港式茶餐厅、上海本帮菜、东北菜馆、粤式烧腊等。
   - 给出你的判断理由（参考店名、菜品关键词、地理位置、价格带等）。

2. **本地竞争格局**
   - 根据竞争对手列表，归类他们的大致菜系（如：Mr Szechuan = 川菜，Khao Tiew = 泰国菜 等）。
   - 对比本店在：评分、评论量、价格带、记忆点（特色菜/招牌）上的优势与短板。

3. **Google 商家资料（GBP）优化建议**
   - 根据 gbp_score 与检查项，给出最优先要补的 3–5 项（如：照片、营业时间、服务选项等）。
   - 每项都写出：具体要做什么 + 这件事如何帮助提高曝光/点击/下单。

4. **网站内容与转化建议**
   - 结合 website_score、字数与文本片段，评价目前网站在：
     - 是否讲清楚菜系与招牌菜
     - 是否有足够内容支撑 SEO
     - 是否有清晰的在线下单/订位 CTA
   - 给出 3–5 条具体优化建议（增加什么板块、需要出现哪些关键词、是否要增加套餐/团体菜单等）。

5. **堂食 & 外卖收入增长策略**
   - 已知堂食客单价约 {dine_in_aov} 美元、外卖客单价约 {delivery_aov} 美元。
   - 设计 3 套组合打法，每套说明：
     - 主攻人群（家庭聚餐、办公室午餐、学生夜宵等）
     - 在 Google/官网/第三方外卖平台上分别要做的动作
     - 预期带来的变化（如：Google 点击提升、外卖复购提升等）。

请用小标题 + 列表形式输出，语气务实、接地气，面向湾区/北美华人餐厅老板。
"""

def call_llm_safe(messages):
    try:
        # 首选：gpt-4.1-mini
        return client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.4,
        ).choices[0].message.content

    except Exception:
        # 没权限则 fallback
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
        ).choices[0].message.content

completion_text = call_llm_safe([
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_msg},
])
return completion_text
# =========================
# 主界面：1 地址 → 候选餐厅
# =========================

st.markdown("## 1️⃣ 输入餐厅地址（自动匹配附近餐厅）")

address_input = st.text_input(
    "餐厅地址（例如：1115 Clement St, San Francisco, CA）",
    "",
    help="可以是完整地址或街道 + 城市，系统会用 Google 自动匹配附近的餐厅。",
)

if st.button("🔍 根据地址查找附近餐厅"):
    if not address_input.strip():
        st.error("请先输入地址。")
    else:
        with st.spinner("根据地址定位并查找附近餐厅..."):
            geocode_res = google_geocode(GOOGLE_API_KEY, address_input)
            if not geocode_res:
                st.error("无法通过该地址找到位置，请检查拼写。")
            else:
                loc = geocode_res[0]["geometry"]["location"]
                lat = loc["lat"]
                lng = loc["lng"]
                nearby = google_places_nearby(
                    GOOGLE_API_KEY, lat, lng, radius_m=300, type_="restaurant"
                )
                if not nearby:
                    st.warning("附近 300 米内未找到餐厅，请尝试输入更精确的地址或放大范围。")
                else:
                    st.session_state["candidate_places"] = nearby
                    st.session_state["run_analysis"] = False
                    st.success(f"已找到 {len(nearby)} 家附近餐厅，请在下方选择你的餐厅。")

# =========================
# 2 选择餐厅 + 业务参数
# =========================

candidate_places = st.session_state["candidate_places"]
selected_place_id: Optional[str] = None
place_label_list: List[str] = []

if candidate_places:
    st.markdown("### 选择你的餐厅")

    for p in candidate_places:
        label = f"{p.get('name', 'Unnamed')} — {p.get('vicinity', '')}"
        place_label_list.append(label)

    selected_index = st.selectbox(
        "在附近餐厅列表中选择你要分析的那一家：",
        options=list(range(len(place_label_list))),
        format_func=lambda i: place_label_list[i],
        index=st.session_state.get("selected_index", 0),
    )
    st.session_state["selected_index"] = selected_index
    selected_place_id = candidate_places[selected_index]["place_id"]

    st.markdown("### 填写业务参数")

    col_aov1, col_aov2 = st.columns(2)
    with col_aov1:
        dine_in_aov = st.number_input(
            "堂食平均客单价（USD）",
            min_value=5.0,
            max_value=300.0,
            value=35.0,
            step=1.0,
        )
    with col_aov2:
        delivery_aov = st.number_input(
            "外卖平均客单价（USD）",
            min_value=5.0,
            max_value=300.0,
            value=45.0,
            step=1.0,
        )

    st.markdown("### 关键词 & 搜索量（不懂就用默认值）")

    keywords_input = st.text_input(
        "核心关键词（逗号分隔）",
        "best chinese food, best asian food, best baked chicken",
        help="用于估算你在 Google 本地搜索里的机会。不懂就用默认值。",
    )

    monthly_search_volume = st.number_input(
        "估算每个核心关键词的月搜索量（统一粗略值）",
        min_value=50,
        max_value=50000,
        value=500,
        step=50,
        help="简单理解为：这一类关键词大概每月有多少人搜索。",
    )

    website_override = st.text_input(
        "如果你的官网和 Google 里记录的不一样，在这里填你的官网 URL（可选）",
        "",
    )

    if st.button("🚀 运行分析"):
        st.session_state["run_analysis"] = True
else:
    st.info("先输入地址并点击“根据地址查找附近餐厅”。")

# =========================
# 3 主分析逻辑
# =========================

if candidate_places and selected_place_id and st.session_state["run_analysis"]:
    # 1. 详情
    with st.spinner("获取餐厅详情（Google Place Details）..."):
        place_detail = google_place_details(GOOGLE_API_KEY, selected_place_id)

    st.success(f"已锁定餐厅：**{place_detail.get('name', 'Unknown')}**")

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
        st.write("**官网（Google）**:", place_detail.get("website", "N/A"))

    geometry = place_detail.get("geometry", {}).get("location", {})
    lat = geometry.get("lat")
    lng = geometry.get("lng")

    # 2. 附近竞争对手
    st.markdown("## 2️⃣ 附近竞争对手（3 公里范围）")
    competitors_df = None
    competitors = []

    if lat is None or lng is None:
        st.warning("未能从 Google 获取经纬度，无法搜索附近竞争对手。")
    else:
        with st.spinner("搜索附近餐厅作为竞争对手..."):
            competitors = google_places_nearby(
                GOOGLE_API_KEY, lat, lng, radius_m=3000, type_="restaurant"
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
            competitors_df = pd.DataFrame(comp_data)
            competitors_df = competitors_df[
                competitors_df["Name"].str.lower()
                != place_detail.get("name", "").lower()
            ]
            competitors_df = competitors_df.sort_values(
                by=["Rating", "Reviews"], ascending=[False, False]
            ).reset_index(drop=True)
            st.dataframe(competitors_df, use_container_width=True)
        else:
            st.info("未找到竞争对手（可能 API 限制或附近餐厅很少）。")

    # 3. GBP 评分
    st.markdown("## 3️⃣ Google 商家资料评分（40 分制）")
    gbp_result = score_gbp_profile(place_detail)
    st.metric("Google 商家资料得分", f"{gbp_result['score']} / 40")

    gbp_rows = []
    for label, (pts, ok) in gbp_result["checks"].items():
        gbp_rows.append(
            {"检查项": label, "得分": pts, "状态": "✅ 完成" if ok else "❌ 缺失/不完整"}
        )
    st.table(pd.DataFrame(gbp_rows))

    # 4. 网站评分
    st.markdown("## 4️⃣ 网站内容 & 体验评分（40 分制）")
    effective_website = website_override or place_detail.get("website")

    if not effective_website:
        st.warning("未提供网站 URL，也无法从 Google 获取，网站评分为 0。")
        web_result = {
            "score": 0,
            "checks": {"无网站": (0, False)},
            "word_count": 0,
            "title": "",
            "text_snippet": "",
        }
    else:
        with st.spinner(f"抓取网站：{effective_website}"):
            html = fetch_html(effective_website)
        web_result = score_website_basic(effective_website, html)

    st.metric("网站得分", f"{web_result['score']} / 40")
    web_rows = []
    for label, (pts, ok) in web_result["checks"].items():
        web_rows.append(
            {"检查项": label, "得分": pts, "状态": "✅ 是" if ok else "❌ 否"}
        )
    st.table(pd.DataFrame(web_rows))

    # 5. 关键词排名 & 堂食/外卖营收损失
    st.markdown("## 5️⃣ 关键词排名 & 堂食 / 外卖潜在营收损失")

    keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
    rank_results: List[Dict[str, Any]] = []

    if not keywords:
        st.info("未提供关键词，跳过排名模拟和营收估算。")
    else:
        for kw in keywords:
            st.write(f"### 关键词：**{kw}**")
            rank_bucket = "none"
            rank_position = None

            if SERPAPI_KEY and lat is not None and lng is not None:
                with st.spinner(f"使用 SerpAPI 查询 Google Maps 排名：{kw}"):
                    try:
                        serp_json = serpapi_google_maps_search(
                            SERPAPI_KEY, kw, lat, lng
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
                # 没有 SerpAPI：用评分+评论简单近似排序
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
                    positions = df_all["name"].str.lower().tolist()
                    name_lower = place_detail.get("name", "").lower()
                    if name_lower in positions:
                        pos = positions.index(name_lower) + 1
                        rank_position = pos
                        if pos <= 3:
                            rank_bucket = "top3"
                        elif pos <= 10:
                            rank_bucket = "4-10"
                        else:
                            rank_bucket = "none"

            monthly_loss_dine_in = estimate_revenue_loss(
                monthly_search_volume,
                rank_bucket,
                dine_in_aov,
                channel="dine-in",
            )
            monthly_loss_delivery = estimate_revenue_loss(
                monthly_search_volume,
                rank_bucket,
                delivery_aov,
                channel="delivery",
            )

            st.write(
                f"- 估计当前排名："
                f"{'Top 3' if rank_bucket=='top3' else ('第 4–10 名' if rank_bucket=='4-10' else '未进入前 10')}"
                f"{'' if rank_position is None else f'（推测名次：{rank_position}）'}"
            )
            st.write(
                f"- 堂食：每月可能少赚约 **${monthly_loss_dine_in:,.0f}**；"
                f"外卖：每月可能少赚约 **${monthly_loss_delivery:,.0f}**。"
            )

            rank_results.append(
                {
                    "关键词": kw,
                    "预估名次": rank_position,
                    "名次区间": rank_bucket,
                    "堂食月损失($)": round(monthly_loss_dine_in, 2),
                    "外卖月损失($)": round(monthly_loss_delivery, 2),
                }
            )

        if rank_results:
            st.markdown("#### 关键词 & 堂食/外卖营收损失汇总")
            st.dataframe(pd.DataFrame(rank_results), use_container_width=True)

    # 6. 综合得分
    st.markdown("## 6️⃣ 总体在线健康总结")
    total_score = gbp_result["score"] + web_result["score"]
    st.metric("综合得分（Profile + Website）", f"{total_score} / 80")
    st.write(
        "- **40 分以下**：在线基础非常弱，基本属于 “Poor”。\n"
        "- **40–60 分**：中等，能被找得到，但不占优势。\n"
        "- **60 分以上**：相对健康，可以开始玩精细化运营和活动。\n"
    )

    # 7. ChatGPT 深度分析
    st.markdown("## 7️⃣ ChatGPT 多维菜系 & 运营分析")

    if not OPENAI_API_KEY:
        st.warning("未配置 OPENAI_API_KEY，如需 AI 深度分析请在 Secrets 中添加。")
    else:
        if st.button("🤖 生成 AI 深度分析报告"):
            with st.spinner("正在调用 ChatGPT 分析..."):
                try:
                    ai_report = llm_deep_analysis(
                        place_detail=place_detail,
                        gbp_result=gbp_result,
                        web_result=web_result,
                        competitors_df=competitors_df,
                        rank_results=rank_results,
                        monthly_search_volume=monthly_search_volume,
                        dine_in_aov=dine_in_aov,
                        delivery_aov=delivery_aov,
                    )
                    st.markdown(ai_report)
                except Exception as e:
                    st.error(f"调用 ChatGPT API 出错：{e}")

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

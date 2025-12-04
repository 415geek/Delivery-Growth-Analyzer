import os
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
    page_title="Restaurant Local SEO & Competitor Analyzer",
    layout="wide",
)

# ---- 从 Streamlit Secrets 读取 API 密钥 ----
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
YELP_API_KEY = st.secrets.get("YELP_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

if not GOOGLE_API_KEY:
    st.error("缺少 GOOGLE_API_KEY，请先在 Streamlit Secrets 中配置后再刷新。")
    st.stop()

# 配置 OpenAI 客户端（可选）
client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# 页面标题
# =========================
st.title("🍜 Restaurant Local SEO & Competitor Analyzer")
st.write(
    "复刻 Owner.com 风格的餐厅在线健康检测 + ChatGPT 深度分析：\n"
    "- 自动识别附近竞争对手\n"
    "- 评估 Google 商家资料完整度（40分）\n"
    "- 检查网站基础 SEO / 内容（40分）\n"
    "- 模拟本地搜索排名 + 粗略营收损失\n"
    "- 使用 ChatGPT 对菜系细分（粤/川/沪/东北/茶餐厅等）与运营给出多维分析"
)

# =========================
# 侧边栏：业务参数（非密钥）
# =========================
st.sidebar.header("📊 分析参数")

default_radius_km = st.sidebar.slider(
    "竞争对手搜索半径（公里）", 0.5, 10.0, 3.0, 0.5
)

avg_order_value = st.sidebar.number_input(
    "平均客单价（USD）", min_value=5.0, max_value=200.0, value=40.0, step=1.0
)
assumed_ctr = st.sidebar.slider(
    "点击率假设（用户看到你后会点进资料/网站的比例）",
    0.05, 0.5, 0.15, 0.01
)
assumed_conv = st.sidebar.slider(
    "下单转化率假设（点进来后下单的比例）",
    0.05, 0.5, 0.20, 0.01
)

st.sidebar.caption("上面三项只用于粗略估算潜在营收损失，可根据实际调整。")

# =========================
# 工具函数（带缓存）
# =========================

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

    # 全站文本
    texts = soup.get_text(separator=" ", strip=True)
    word_count = len(texts.split())
    text_snippet = texts[:3000]  # 传给 ChatGPT 用

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
    keywords = ["chinese", "cantonese", "szechuan", "sichuan", "shanghai",
                "noodle", "rice", "dumpling", "hot pot", "bbq", "dim sum"]
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
    ctr: float,
    conv: float,
) -> float:
    """粗略营收损失估算。"""
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
    avg_order_value: float,
) -> str:
    if client is None:
        return "未配置 OPENAI_API_KEY，无法调用 ChatGPT。"

    # 只取前 5 个竞争对手，避免 prompt 太长
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
            "avg_order_value": avg_order_value,
        },
    }

    text_snippet = web_result.get("text_snippet", "")

    system_msg = (
        "你是一名专门服务北美餐馆的本地营销和外卖运营顾问，曾任职于麦肯锡一个专门做餐饮分析的部门"
        "非常了解世界各地的菜系，尤其在中餐菜系的细分领域属于行业权威，如粤菜、茶餐厅、川菜、湘菜、东北菜、上海菜等细分菜系，"
        "熟悉 Google 本地搜索和 UberEats/DoorDash/Grubhub/Hungrypanda/Fantuan 等平台的运营逻辑。"
        "请用简体中文回答，但在需要时可加少量英文术语。"
    )

    user_msg = f"""
这是一个餐厅的在线数据，请你做**多维深度分析**并给出细分菜系判断与运营建议。

【结构化数据 JSON】
{json.dumps(payload, ensure_ascii=False, indent=2)}

【网站文本片段（最多 3000 字符）】
{text_snippet}

请你完成以下任务：

1. **菜系细分判断**
   - 根据餐厅名称、网站文本、Google 类型等，判断它更像：粤菜馆？茶餐厅？川菜？东北菜？上海菜？融合亚洲？其他？
   - 给出 1–2 句理由，并用中文给出 1 个最合适的菜系标签（如：`正宗川菜馆`、`港式茶餐厅`）。

2. **本地竞争格局分析**
   - 根据竞争对手列表，判断：他们主要是哪几类餐厅（例：Mr Szechuan = 川菜，Khao Tiew = 泰国菜等）。
   - 简要说明：当前这家餐厅在“价格带、评分、评论量、品牌记忆点”上相比竞争对手的优势和劣势。

3. **Google 商家资料优化建议（GBP）**
   - 根据 gbp_score 和 checks，列出最优先需要补齐的 3–5 项（例如：上传更多菜品照片、补充营业时间、增加服务选项等）。
   - 对每一项给出具体执行建议（要写得像你要跟老板解释，“为什么做这件事会多带来订单”）。

4. **网站内容与转化建议**
   - 根据 website_score、word_count 和网站文本片段，指出目前网站内容在以下几个维度是否达标：
     - 是否清晰说明菜系和招牌菜？
     - 是否有足够文本支撑 SEO？
     - 是否有强的在线下单/预订 CTA？
   - 给出 3–5 条具体建议，包含：应该增加什么板块（例如：招牌菜介绍、午市套餐、家庭聚会/宴会页面等）、需要加入哪些关键词。

5. **外卖与本地搜索增长策略**
   - 结合 rank_results 的关键词和你对菜系的判断，给出 3 条“攻占 Google 搜索 + 外卖平台”的组合打法。
   - 每条打法都要包含：
     - 目标关键词（中英都可以）
     - 在 Google 商家、网站、外卖平台各自要做什么调整
     - 预期会带来怎样类型的客人（家庭聚餐、办公室午餐、学生夜宵等）。

要求：
- 用小标题 + 列表的方式输出，方便复制到报告里。
- 语气专业但接地气，面向湾区/北美华人餐厅老板。
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",   # 或 gpt-4o-mini / gpt-4.1，看你账号权限
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
    )

    return completion.choices[0].message.content


# =========================
# 主界面交互
# =========================

st.markdown("## 1️⃣ 输入餐厅信息")

col1, col2 = st.columns(2)
with col1:
    restaurant_name = st.text_input("餐厅名称（Restaurant Name）", "")
with col2:
    city_region = st.text_input("城市/区域（如：Outer Sunset, San Francisco）", "")

website_url = st.text_input(
    "餐厅官网 URL（可留空，优先使用 Google 商家里记录的官网）",
    "",
)

keywords_input = st.text_input(
    "核心关键词（逗号分隔，例如：best chinese food outer sunset, best asian food west portal）",
    "best chinese food outer sunset, best asian food outer sunset",
)

monthly_search_volume = st.number_input(
    "估算每个核心关键词的月搜索量（统一粗略值）",
    min_value=50,
    max_value=50000,
    value=500,
    step=50,
)

run_btn = st.button("🚀 运行分析")

if run_btn:
    if not restaurant_name or not city_region:
        st.error("请填写餐厅名称和城市/区域。")
        st.stop()

    query = f"{restaurant_name} {city_region}"
    with st.spinner(f"在 Google Places 中搜索：{query}"):
        places = google_places_search(GOOGLE_API_KEY, query)

    if not places:
        st.error("Google Places 未找到匹配餐厅，请检查名称和城市。")
        st.stop()

    target = places[0]
    place_id = target["place_id"]

    with st.spinner("获取餐厅详情（Google Place Details）..."):
        place_detail = google_place_details(GOOGLE_API_KEY, place_id)

    st.success(f"已找到餐厅：**{place_detail.get('name', 'Unknown')}**")

    # ---- 基础信息 ----
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

    # ---- 竞争对手 ----
    st.markdown("## 2️⃣ 附近竞争对手（Google Places Nearby）")
    competitors_df = None
    competitors = []

    if lat is None or lng is None:
        st.warning("未能从 Google 获取经纬度，无法搜索附近竞争对手。")
    else:
        radius_m = int(default_radius_km * 1000)
        with st.spinner("搜索附近餐厅作为竞争对手..."):
            competitors = google_places_nearby(
                GOOGLE_API_KEY, lat, lng, radius_m, type_="restaurant"
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
            st.info("未找到竞争对手（可能半径太小或 API 限制）。")

    # ---- GBP 评分 ----
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
    st.table(pd.DataFrame(gbp_rows))

    # ---- 网站评分 ----
    st.markdown("## 4️⃣ 网站内容 & 体验评分（40 分制）")
    effective_website = website_url or place_detail.get("website")

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
            {
                "检查项": label,
                "得分": pts,
                "状态": "✅ 是" if ok else "❌ 否",
            }
        )
    st.table(pd.DataFrame(web_rows))

    # ---- 关键词排名 + 营收损失 ----
    st.markdown("## 5️⃣ 本地关键词排名 & 潜在营收损失")

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
                # 没有 SerpAPI：用评分+评论简单近似排序，模拟本地排名
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
            st.dataframe(pd.DataFrame(rank_results), use_container_width=True)

    # ---- 综合得分 ----
    st.markdown("## 6️⃣ 总体在线健康总结")
    total_score = gbp_result["score"] + web_result["score"]
    st.metric("综合得分（Profile + Website）", f"{total_score} / 80")
    st.write(
        "- **40 分以下**：在线基础非常弱，基本属于 “Poor”。\n"
        "- **40–60 分**：中等，能被找得到，但不占优势。\n"
        "- **60 分以上**：相对健康，可以开始玩精细化运营和活动。\n"
    )

    # =========================
    # ChatGPT 深度多维分析
    # =========================
    st.markdown("## 7️⃣ ChatGPT 多维菜系 & 运营分析")

    if not OPENAI_API_KEY:
        st.warning("未配置 OPENAI_API_KEY，如需 AI 深度分析请在 Secrets 中添加。")
    else:
        if st.button("🤖 生成 AI 深度分析（细分菜系 + 运营建议）"):
            with st.spinner("正在调用 ChatGPT 分析，请稍候..."):
                try:
                    ai_report = llm_deep_analysis(
                        place_detail=place_detail,
                        gbp_result=gbp_result,
                        web_result=web_result,
                        competitors_df=competitors_df,
                        rank_results=rank_results,
                        monthly_search_volume=monthly_search_volume,
                        avg_order_value=avg_order_value,
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

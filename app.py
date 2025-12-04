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

# 尝试可选导入 headless 浏览器支持
try:
    from requests_html import HTMLSession  # 可能在某些环境缺依赖
    HAS_REQUESTS_HTML = True
except Exception:
    HTMLSession = None
    HAS_REQUESTS_HTML = False

# =========================
# 基本配置 & Secrets
# =========================
st.set_page_config(
    page_title="Aurainsight 餐馆增长诊断",
    layout="wide",
)

st.title("Aurainsight 餐馆增长诊断")
st.write(
    "针对北美餐馆老板的一键在线体检：\n"
    "- 只需输入地址，自动匹配你的餐厅\n"
    "- 自动抓取附近竞争对手\n"
    "- 估算堂食 / 外卖的潜在流失营收\n"
    "- 尝试抓取官网 / 第三方平台菜单，结合作品级 ChatGPT 报告做多维菜系 & 菜单结构分析"
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
# Session State 初始化
# =========================
if "candidate_places" not in st.session_state:
    st.session_state["candidate_places"] = []
if "selected_index" not in st.session_state:
    st.session_state["selected_index"] = 0

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
    先尝试带 fields，如果 SDK/版本不支持就 fallback 到不带 fields 的调用。
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
        "url",
    ]
    try:
        result = gmaps.place(place_id=place_id, fields=fields)
        data = result.get("result", result)
    except Exception:
        result = gmaps.place(place_id=place_id)
        data = result.get("result", result)
    return data


@st.cache_data(show_spinner=False)
def google_places_nearby(
    api_key: str, lat: float, lng: float, radius_m: int, type_: str = "restaurant"
) -> List[Dict[str, Any]]:
    gmaps = gm_client(api_key)
    result = gmaps.places_nearby(location=(lat, lng), radius=radius_m, type=type_)
    return result.get("results", [])


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


@st.cache_data(show_spinner=False)
def fetch_html(url: str) -> Optional[str]:
    """
    先用普通 requests 抓一次；
    如果失败，并且环境支持 requests_html，再尝试 headless 渲染。
    Streamlit Cloud 上如果缺 lxml 相关依赖，会自动关闭 headless，不会报错。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
    }

    # 第一次尝试：普通 HTTP 请求
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code < 400 and "text/html" in resp.headers.get("Content-Type", ""):
            return resp.text
        st.warning(f"[菜单抓取] 普通请求效果一般，状态码 {resp.status_code}。")
    except Exception as e:
        st.warning(f"[菜单抓取] 普通请求出错：{e}")

    # 第二次（可选）尝试：headless 浏览器执行 JS
    if not HAS_REQUESTS_HTML:
        # 当前环境不支持 headless，就直接结束
        st.info("当前运行环境不支持 headless 浏览器渲染，已退回普通抓取模式。")
        return None

    try:
        session = HTMLSession()
        r = session.get(url, headers=headers, timeout=30)
        r.html.render(timeout=40, sleep=2)
        return r.html.html
    except Exception as e:
        st.warning(f"[菜单抓取] headless 渲染失败：{e}")
        return None

# =========================
# 评分 & 计算函数
# =========================

def score_gbp_profile(place: Dict[str, Any]) -> Dict[str, Any]:
    """简化版 Google 商家资料评分，总分 40 分。"""
    score = 0
    checks: Dict[str, Any] = {}

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
    has_hours = bool(opening_hours.get("weekday_text")) or opening_hours.get("open_now") is not None
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
    checks: Dict[str, Any] = {}

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
# 菜单相关
# =========================

def extract_menu_text_from_html(html: str) -> str:
    """从 HTML 中尽量提取出像菜单的内容（菜名 + 价格等）"""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    texts = []
    for el in soup.find_all(["h2", "h3", "h4", "li", "p", "span", "div"]):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if any(x in txt for x in ["$", "¥"]) or any(
            kw in txt.lower()
            for kw in ["chicken", "beef", "pork", "noodle", "rice", "tofu", "dumpling", "soup"]
        ):
            if 3 <= len(txt) <= 120:
                texts.append(txt)

    if not texts:
        full = soup.get_text(" ", strip=True)
        return full[:4000]

    seen = set()
    deduped = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    return "\n".join(deduped[:400])


def build_menu_payload(menu_urls: List[str]) -> List[Dict[str, str]]:
    menus: List[Dict[str, str]] = []
    for url in menu_urls:
        url = url.strip()
        if not url:
            continue

        html = fetch_html(url)
        if not html:
            menus.append(
                {
                    "source": urlparse(url).netloc or "unknown",
                    "url": url,
                    "status": "fetch_failed_or_blocked",
                    "menu_text": "",
                }
            )
            continue

        menu_text = extract_menu_text_from_html(html)
        status = "ok" if menu_text.strip() else "no_menu_detected"

        menus.append(
            {
                "source": urlparse(url).netloc or "unknown",
                "url": url,
                "status": status,
                "menu_text": menu_text,
            }
        )

    return menus


def discover_menu_urls(place_detail: Dict[str, Any], website_html: Optional[str]) -> List[str]:
    """
    尝试自动发现菜单/点餐链接：
    - 自家官网
    - 官网页面里包含 menu/order 的链接
    - 常见第三方外卖平台链接
    """
    urls = set()

    main_site = place_detail.get("website")
    if main_site:
        urls.add(main_site)

    if "url" in place_detail:
        urls.add(place_detail["url"])

    if website_html:
        soup = BeautifulSoup(website_html, "lxml")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            href_lower = href.lower()
            text = a.get_text(" ", strip=True).lower()

            if any(k in href_lower for k in ["menu", "order", "online-order", "order-online"]) or \
               any(k in text for k in ["menu", "order", "online order"]):
                urls.add(href)

            for domain in [
                "doordash.com",
                "ubereats.com",
                "grubhub.com",
                "hungrypanda.co",
                "fantuan.ca",
                "order.online",
                "chownow.com",
            ]:
                if domain in href_lower:
                    urls.add(href)

    return list(urls)

# =========================
# ChatGPT 深度分析函数
# =========================

def call_llm_safe(messages: List[Dict[str, str]]) -> str:
    if client is None:
        return "未配置 OPENAI_API_KEY，无法调用 ChatGPT。"
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.4,
        )
        return completion.choices[0].message.content
    except Exception:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
        )
        return completion.choices[0].message.content


def llm_deep_analysis(
    place_detail: Dict[str, Any],
    gbp_result: Dict[str, Any],
    web_result: Dict[str, Any],
    competitors_df: Optional[pd.DataFrame],
    rank_results: List[Dict[str, Any]],
    monthly_search_volume: int,
    dine_in_aov: float,
    delivery_aov: float,
    menus_payload: List[Dict[str, str]],
) -> str:
    comp_json = []
    if competitors_df is not None and not competitors_df.empty:
        sub = competitors_df.head(6)
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
        "menus": menus_payload,
    }

    text_snippet = web_result.get("text_snippet", "")

    system_msg = (
        "你是一名专门服务北美餐馆的本地营销和外卖运营顾问，曾任职于麦肯锡一个专门做餐饮分析的部门，"
        "非常了解世界各地的菜系，尤其在中餐菜系的细分领域属于行业权威，如粤菜、茶餐厅、川菜、湘菜、东北菜、上海菜等，"
        "熟悉 Google 本地搜索和 UberEats/DoorDash/Grubhub/Hungrypanda/Fantuan 等平台的运营逻辑。"
        "请用简体中文回答，但在需要时可加少量英文术语。"
    )

    user_msg = f"""
这是一个餐厅的在线数据和菜单片段，请你做**多维深度分析**：

【结构化数据 JSON】
{json.dumps(payload, ensure_ascii=False, indent=2)}

【网站文本片段（最多 3000 字符）】
{text_snippet}

请你完成以下任务（分段输出）：

1. **菜系细分判断**
   - 判断该店的主菜系和子菜系（例如：粤菜-茶餐厅、川菜-辣炒、东北家常菜、上海本帮菜等），说明依据。
   - 如果菜单里有多种菜系，请说明主次结构。

2. **菜单结构与价格带分析**
   - 根据菜单文本，分析：
     - 热门品类（如主食类、招牌菜、套餐、炸鸡、甜品等）
     - 人均价位区间、主力价格带（例如：多数主菜集中在 $15–$22）
     - 是否存在明显的“利润杀手”（价格偏低但制作复杂、毛利低的菜）

3. **线上曝光 & 竞争态势解读**
   - 结合 GBP 评分、网站得分、关键词排名结果，判断：
     - 目前在本地搜索中的位置（落后程度、有无机会冲击 Top 3）
     - 和 3–5 家核心竞品相比的明显短板和优势。

4. **外卖平台机会点（如果菜单里出现外卖平台链接）**
   - 根据菜品结构和价格，判断适合重点发力的平台类型（聚合外卖 / 自配送 / 线下堂食引流）。
   - 给出 2–3 个具体可执行的促销活动建议（比如：高毛利品类做 BOGO、午市定价逻辑等）。

5. **接下来 30 天可执行的行动清单**
   - 用清单方式给出 5–8 条“餐馆老板能听懂、能马上执行”的改进建议：
     - Google 资料 & 网站内容优先级
     - 菜单结构和定价优化
     - 外卖活动 & 转化率优化建议

要求：
- 尽量用短句和项目符号，方便餐厅老板阅读和执行。
- 对每条建议，简单说明“为什么这么做有用”（基于数据/经验的逻辑）。
"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    return call_llm_safe(messages)

# =========================
# 1️⃣ 输入地址，锁定餐厅
# =========================

st.markdown("## 1️⃣ 输入餐厅地址（自动匹配附近餐厅）")

address_input = st.text_input(
    "餐厅地址（例如：1115 Clement St, San Francisco, CA）",
    "",
    help="可以是完整地址或街道 + 城市，系统会用 Google 自动匹配附近的餐厅。",
)

search_btn = st.button("🔍 根据地址查找附近餐厅")

if search_btn:
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
                    st.success(f"已找到 {len(nearby)} 家附近餐厅，请在下方选择你的餐厅。")

# =========================
# 2️⃣ 选择餐厅 + 业务参数
# =========================

candidate_places = st.session_state["candidate_places"]
selected_place_id: Optional[str] = None
place_label_list: List[str] = []

if candidate_places:
    st.markdown("## 2️⃣ 选择你的餐厅 & 填写关键业务参数")

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

    run_btn = st.button("🚀 运行分析")

else:
    st.info("先输入地址并点击“根据地址查找附近餐厅”。")
    run_btn = False

# =========================
# 3️⃣ 主分析逻辑
# =========================

if candidate_places and selected_place_id and run_btn:
    with st.spinner("获取餐厅详情（Google Place Details）..."):
        place_detail = google_place_details(GOOGLE_API_KEY, selected_place_id)

    st.success(f"已锁定餐厅：**{place_detail.get('name', 'Unknown')}**")

    geometry = place_detail.get("geometry", {})
    location = geometry.get("location", {})
    center_lat = location.get("lat")
    center_lng = location.get("lng")

    with st.spinner("扫描附近 1.5 公里内的竞争对手..."):
        nearby_comp = google_places_nearby(
            GOOGLE_API_KEY, center_lat, center_lng, radius_m=1500, type_="restaurant"
        )

    competitors_rows = []
    for r in nearby_comp:
        pid = r.get("place_id")
        if pid == selected_place_id:
            continue
        competitors_rows.append(
            {
                "name": r.get("name"),
                "vicinity": r.get("vicinity"),
                "rating": r.get("rating"),
                "reviews": r.get("user_ratings_total"),
                "place_id": pid,
            }
        )

    competitors_df = pd.DataFrame(competitors_rows).sort_values(
        by=["rating", "reviews"], ascending=[False, False]
    )

    gbp_result = score_gbp_profile(place_detail)

    website_url = website_override.strip() or place_detail.get("website", "")
    website_html = None
    if website_url:
        with st.spinner("抓取官网页面用于分析..."):
            website_html = fetch_html(website_url)

    web_result = score_website_basic(website_url, website_html)

    st.markdown("## 3️⃣ 关键词排名 & 潜在营收损失（粗略估算）")

    kw_list = [k.strip() for k in keywords_input.split(",") if k.strip()]
    rank_rows: List[Dict[str, Any]] = []

    if SERPAPI_KEY and center_lat and center_lng:
        with st.spinner("通过 SerpAPI 查询 Google Maps 排名..."):
            for kw in kw_list:
                try:
                    serp_json = serpapi_google_maps_search(
                        SERPAPI_KEY, kw, center_lat, center_lng
                    )
                    rank = infer_rank_from_serpapi(serp_json, place_detail.get("name", ""))
                except Exception:
                    rank = None

                if rank is None:
                    bucket = "none"
                elif rank <= 3:
                    bucket = "top3"
                elif rank <= 10:
                    bucket = "4-10"
                else:
                    bucket = "11+"

                dine_loss = estimate_revenue_loss(
                    monthly_search_volume, bucket, dine_in_aov, channel="dine-in"
                )
                delivery_loss = estimate_revenue_loss(
                    monthly_search_volume, bucket, delivery_aov, channel="delivery"
                )

                rank_rows.append(
                    {
                        "关键词": kw,
                        "预估名次": rank,
                        "名次区间": bucket,
                        "堂食月损失($)": round(dine_loss, 1),
                        "外卖月损失($)": round(delivery_loss, 1),
                    }
                )
    else:
        st.warning("未配置 SERPAPI_KEY，无法自动查询 Google Maps 排名，仅展示关键词列表。")
        for kw in kw_list:
            rank_rows.append(
                {
                    "关键词": kw,
                    "预估名次": None,
                    "名次区间": "unknown",
                    "堂食月损失($)": None,
                    "外卖月损失($)": None,
                }
            )

    rank_df = pd.DataFrame(rank_rows)
    st.dataframe(rank_df, use_container_width=True)

    st.markdown("## 4️⃣ Google 商家资料健康状况（Profile）")

    st.write(f"**Profile 评分：{gbp_result['score']} / 40**")
    gbp_checks_df = pd.DataFrame(
        [
            {"检查项": name, "得分": pts, "是否达标": "✅ 是" if ok else "❌ 否"}
            for name, (pts, ok) in gbp_result["checks"].items()
        ]
    )
    st.dataframe(gbp_checks_df, use_container_width=True)

    st.markdown("## 5️⃣ 官网内容 & 结构健康状况（Website）")

    st.write(f"**网站评分：{web_result['score']} / 40**")
    web_checks_df = pd.DataFrame(
        [
            {"检查项": name, "得分": pts, "是否达标": "✅ 是" if ok else "❌ 否"}
            for name, (pts, ok) in web_result["checks"].items()
        ]
    )
    st.dataframe(web_checks_df, use_container_width=True)

    if website_url:
        st.write(f"官网：{website_url}")
    else:
        st.warning("未在 Google 资料中发现官网链接，网站评分会偏低。")

    st.markdown("## 6️⃣ 附近竞争对手概览")

    if not competitors_df.empty:
        st.dataframe(
            competitors_df[["name", "vicinity", "rating", "reviews"]],
            use_container_width=True,
        )
    else:
        st.info("未能找到足够的竞争对手数据。")

    st.markdown("## 7️⃣ 总体在线健康总结")

    total_score = gbp_result["score"] + web_result["score"]
    st.write(f"**综合得分（Profile + Website）：{total_score} / 80**")

    st.write(
        "- 40 分以下：在线基础非常薄弱，基本属于 “Poor”。\n"
        "- 40–60 分：中等，能被找到，但不占优势。\n"
        "- 60 分以上：相对健康，可以开始玩精细化运营和活动。"
    )

    st.markdown("## 8️⃣ ChatGPT 多维菜系 & 菜单结构 & 运营分析")

    auto_menu_urls = discover_menu_urls(place_detail, website_html)
    auto_menu_urls_str = "\n".join(auto_menu_urls)

    st.markdown("#### 菜单抓取预览（可手动增删链接）")
    menu_urls_input = st.text_area(
        "系统自动发现的菜单/点餐链接（每行一个，可自行增删）",
        auto_menu_urls_str,
        height=140,
    )

    menu_urls = [u.strip() for u in menu_urls_input.splitlines() if u.strip()]
    menus_payload: List[Dict[str, str]] = []

    if menu_urls:
        with st.spinner("尝试抓取菜单文本（官网 / 外卖平台）..."):
            menus_payload = build_menu_payload(menu_urls)

        if menus_payload:
            menu_preview_df = pd.DataFrame(
                [
                    {
                        "来源": m["source"],
                        "URL": m["url"],
                        "状态": m["status"],
                        "菜单文本预览": (m["menu_text"] or "")[:120].replace("\n", " "),
                    }
                    for m in menus_payload
                ]
            )
            st.dataframe(menu_preview_df, use_container_width=True)
    else:
        st.info("当前没有可用的菜单链接，AI 分析将主要基于 Google 资料和官网内容。")

    st.markdown("### 🔍 生成 ChatGPT 菜系 & 菜单 & 运营深度分析报告")

    ai_btn = st.button("✨ 生成 AI 深度分析报告")

    if ai_btn:
        with st.spinner("正在调用 ChatGPT 生成分析报告，大概需要几秒钟..."):
            try:
                ai_report = llm_deep_analysis(
                    place_detail=place_detail,
                    gbp_result=gbp_result,
                    web_result=web_result,
                    competitors_df=competitors_df,
                    rank_results=rank_rows,
                    monthly_search_volume=monthly_search_volume,
                    dine_in_aov=dine_in_aov,
                    delivery_aov=delivery_aov,
                    menus_payload=menus_payload,
                )
                st.markdown(ai_report)
            except Exception as e:
                st.error(f"调用 ChatGPT 失败：{e}")

    st.markdown("## 9️⃣ 免费获取完整诊断报告 & 1 对 1 咨询")

    st.markdown(
        """
        <a href="https://wa.me/6289995610" target="_blank"
           style="
             display:inline-block;
             padding:12px 24px;
             background:#25D366;
             color:#ffffff;
             border-radius:8px;
             text-decoration:none;
             font-weight:600;
             font-size:16px;
             margin-top:8px;
           ">
           📲 免费获取完整诊断报告（WhatsApp）
        </a>
        """,
        unsafe_allow_html=True,
    )

# ========== 署名（LinkedIn） ==========
LINKEDIN_URL = "https://www.linkedin.com/in/lingyu-maxwell-lai"

st.markdown(
    f"""
<div style="display:flex;align-items:center;gap:10px;margin-top:18px;margin-bottom:8px;">
  <div style="font-size:14px;color:#666;">
    Builded by <strong>Maxwell Lai</strong>
  </div>
  <a href="{LINKEDIN_URL}" target="_blank" title="LinkedIn: Maxwell Lai"
     style="display:inline-flex;align-items:center;justify-content:center;
            width:18px;height:18px;border-radius:4px;background:#0A66C2;">
    <img src="https://cdn.jsdelivr.net/gh/simple-icons/simple-icons/icons/linkedin.svg"
         alt="LinkedIn" width="12" height="12" style="filter: invert(1);" />
  </a>
</div>
""",
    unsafe_allow_html=True,
)

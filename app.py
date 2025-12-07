import os
import json
import base64
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
    "- 自动扫描附近竞争对手\n"
    "- 估算堂食 / 外卖的潜在流失营收\n"
    "- 抓取官网 / 外卖平台菜单 + Google 菜单图片，结合 ChatGPT 做多维菜系 & 菜单结构 & 运营分析\n"
    "- 基于菜单菜系画像，自动筛选真正的核心竞对（实验功能）"
)

# 从 Streamlit Secrets 读取 API 密钥
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
SERPAPI_KEY = st.secrets.get("SERPAPI_KEY", "")
YELP_API_KEY = st.secrets.get("YELP_API_KEY", "")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
SCRAPERAPI_KEY = st.secrets.get("SCRAPERAPI_KEY", "")

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
if "analysis_ready" not in st.session_state:
    st.session_state["analysis_ready"] = False
if "ocr_menu_texts" not in st.session_state:
    st.session_state["ocr_menu_texts"] = []

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


# =========================
# ScraperAPI 集成
# =========================

@st.cache_data(show_spinner=False)
def fetch_html_via_scraperapi(url: str, render: bool = True) -> Optional[str]:
    """
    通过 ScraperAPI 抓取页面，自动绕过大部分反爬 & Cloudflare。
    render=True 会启用 JS 渲染，适合 order.online / Doordash 这类 SPA。
    """
    if not SCRAPERAPI_KEY:
        return None

    api_endpoint = "https://api.scraperapi.com"
    params = {
        "api_key": SCRAPERAPI_KEY,
        "url": url,
    }
    if render:
        params["render"] = "true"

    try:
        resp = requests.get(api_endpoint, params=params, timeout=40)
        resp.raise_for_status()
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" in ctype or "application/json" in ctype:
            return resp.text
        return None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def fetch_html(url: str) -> Optional[str]:
    """
    统一页面抓取逻辑：
    1）遇到典型强 JS/反爬域名（Doordash/order.online 等）优先走 ScraperAPI；
    2）普通请求试一次；
    3）失败再走 ScraperAPI；
    4）再失败用本地 headless（requests_html）兜底。
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
    }

    hard_domains = [
        "doordash.com",
        "ubereats.com",
        "grubhub.com",
        "order.online",
        "hungrypanda.co",
        "fantuan.ca",
        "chownow.com",
    ]
    lower_url = url.lower()

    # 0️⃣ 某些第三方点餐网站直接走 ScraperAPI + JS 渲染
    if any(d in lower_url for d in hard_domains):
        html = fetch_html_via_scraperapi(url, render=True)
        if html:
            return html

    # 1️⃣ 普通请求（适合自家官网、简单点餐站）
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        ctype = resp.headers.get("Content-Type", "")
        body = resp.text

        blocked = (
            resp.status_code >= 400
            or "captcha" in body.lower()
            or "access denied" in body.lower()
            or "temporarily blocked" in body.lower()
        )

        if resp.status_code < 400 and "text/html" in ctype and not blocked:
            return body
    except Exception:
        pass

    # 2️⃣ 普通请求失败 → ScraperAPI（渲染打开）
    if SCRAPERAPI_KEY:
        html = fetch_html_via_scraperapi(url, render=True)
        if html:
            return html

    # 3️⃣ 再失败 → requests_html headless 渲染（如果可用）
    if not HAS_REQUESTS_HTML:
        return None

    try:
        session = HTMLSession()
        r = session.get(url, headers=headers, timeout=30)
        r.html.render(timeout=40, sleep=2)
        return r.html.html
    except Exception:
        return None

# =========================
# Google 菜单照片 & OCR
# =========================

@st.cache_data(show_spinner=False)
def fetch_place_photo(api_key: str, photo_reference: str, maxwidth: int = 1200) -> bytes:
    """
    调用 Google Place Photos API，返回图片二进制。
    """
    url = "https://maps.googleapis.com/maps/api/place/photo"
    params = {
        "key": api_key,
        "photoreference": photo_reference,
        "maxwidth": maxwidth,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.content


def classify_menu_image(img_bytes: bytes) -> str:
    """
    使用 GPT 多模态判断图片类型：
    返回：
      - "menu_page"           明显是菜牌/菜单页面
      - "food_dish"           单道菜/几道菜的摆盘照片
      - "storefront_or_other" 店招、Logo、环境、人像等
    """
    if client is None:
        return "storefront_or_other"

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    prompt = """
你是一名餐饮图片识别助手，请只根据图片内容判断图片类型，不要做其他事情。

请从下面三种类型中选一个，并只输出对应的英文代码（不要加解释）：

1. 如果图片主要内容是「菜单/菜牌页面」，特征包括：
   - 有成列的菜品名称、描述和价格
   - 看起来像打印出来的 menu / laminated menu / 手写菜单板
   - 可能是一页或多页菜单的照片
   请输出：menu_page

2. 如果图片主要内容是「一盘或几盘菜、饮品」，特征包括：
   - 看得到实际食物/饮料摆盘
   - 没有成列的菜单条目和价格
   请输出：food_dish

3. 如果图片主要内容是「店招、门面、Logo、环境、人像、街景等」，而不是菜单或菜品特写，
   请输出：storefront_or_other

重要规则：
- 只输出以上三种之一的英文代码，不要输出任何说明文字。
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0.0,
    )
    label = (resp.choices[0].message.content or "").strip().lower()
    if label not in {"menu_page", "food_dish", "storefront_or_other"}:
        return "storefront_or_other"
    return label


def get_place_photos(place_detail: Dict[str, Any], max_photos: int = 20) -> List[Dict[str, Any]]:
    """
    从 Place Details 中获取照片，并自动筛选出“菜单页”优先返回。
    """
    photos = place_detail.get("photos", []) or []
    results: List[Dict[str, Any]] = []
    if not photos:
        return results

    for p in photos[:max_photos]:
        ref = p.get("photo_reference")
        if not ref:
            continue
        try:
            img_bytes = fetch_place_photo(GOOGLE_API_KEY, ref, maxwidth=1000)
        except Exception:
            continue

        label = classify_menu_image(img_bytes)
        if label == "menu_page":
            results.append(
                {
                    "photo_reference": ref,
                    "image_bytes": img_bytes,
                    "label": label,
                }
            )

    return results


def ocr_menu_from_image_bytes(img_bytes: bytes) -> str:
    """
    使用 OpenAI 多模态从图片中提取菜单信息：
    - 如果有菜单文字：输出菜名 + 价格
    - 如果无文字但有菜品照片：猜菜名，价格 unknown
    - 如果只是门头/环境/Logo：返回空字符串（忽略）
    """
    if client is None:
        return ""

    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    prompt = """
你现在要判断这张图片是不是“有用的菜单相关图片”。

请按以下逻辑处理：

1. 如果图片上有明显的菜单文字（例如菜名、描述、价格、类似菜单排版）：
   - 只提取菜品名称和价格。
   - 每行输出一个菜，格式：
     菜名原文 - 英文名(如果有就写，没有就留空) - 价格
   - 如果是多种规格，可以拆成多行。
   - 不要输出任何额外说明。

2. 如果图片上没有明显的文字，但能清楚看到一盘菜或一杯饮料等“单品菜品照片”：
   - 猜测该菜/饮品最可能的中英文名称。
   - 每行输出一个候选，格式：
     猜测菜名(中文，如果你知道) - English name(如果能判断) - unknown
   - 最多输出 1-3 行。
   - 不要输出其他解释。

3. 如果图片主要是店招、Logo、人像、街景、室内环境，没有可识别的菜单文字，也看不清具体菜品：
   - 不要输出任何内容，返回完全空的结果。

总规则：
- 只输出菜单条目文本，不要加标题、说明、前后缀。
- 如果最后判断属于第 3 种情况，就返回空字符串。
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0.1,
    )
    text = resp.choices[0].message.content or ""
    return text.strip()

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
    elif rank_bucket == "11+":
        current_factor = 0.1
    else:  # none / unknown
        current_factor = 0.0

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
# 菜单相关 & 菜系画像
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

# ========== 菜单菜系画像 & 精准竞对辅助函数 ==========

def analyze_menu_profile(menu_text: str) -> Dict[str, Any]:
    """
    用 ChatGPT 根据菜单文本做菜系画像（川菜 / 粤菜 / 港式茶餐厅 / 点心 / 奶茶店等）
    """
    if client is None:
        return {"error": "未配置 OPENAI_API_KEY，无法进行菜系画像分析。"}

    system_prompt = """
你是一名熟悉北美中餐市场的餐饮顾问，专门根据菜单内容给餐厅做画像。

特别规则（很重要）：
- 如果菜单里出现大量“焗饭、焗猪扒饭、意粉（意大利面）、公仔面、菠萝油、多士、三文治”等，
  并且同时有港式奶茶、鸳鸯等饮品，这家店很大概率是【港式茶餐厅】。
- 如果菜单里出现大量“蒸排骨、凤爪、萝卜糕、虾饺、烧卖、肠粉、叉烧包、流沙包”等点心类菜品，
  并以一笼一笼的小份为主，这家店很大概率是【粤式早茶/点心为主的粤菜馆】。
- 如果两者都有，要看哪一类菜品占比更高：
  - 茶餐厅：主食类焗饭/意粉/公仔面/套餐多，点心只是少量补充。
  - 粤菜酒楼/点心店：点心类品种非常多，焗饭/意粉只是少量出现。

- 川菜特征关键词举例：水煮鱼、麻婆豆腐、毛血旺、酸菜鱼、辣子鸡、干锅、冒菜、串串香等。
- 湘菜特征关键词举例：剁椒鱼头、农家小炒肉、手撕包菜、臭豆腐、口味虾等。
- 北方面馆/饺子馆可以包含：饺子、锅贴、手工面、牛肉面、羊肉串、锅包肉等。

输出必须是 JSON，字段如下：
- primary_cuisine: 主菜系，比如 "川菜", "粤菜", "港式茶餐厅", "粤式点心", "面包店", "奶茶店", "其他中餐"
- secondary_cuisines: 可能的次要菜系列表，比如 ["粤菜", "港式茶餐厅"]
- business_type: "正餐" / "快餐" / "手摇饮" / "烘焙甜品"
- price_level: 从 1 到 4, 对应人均大概 $: 1=便宜, 2=中等, 3=偏高, 4=高端
- signature_items: 菜单中你认为最能代表这家店风格的 3-5 个菜品名（用原文）
- competitor_search_keywords: 搜索竞对时建议用的英文关键词列表
- notes: 你的判断依据和提醒（中文）
只输出 JSON。
    """.strip()

    user_prompt = f"以下是这家餐厅的菜单内容（菜名+简介，可以不完整）：\n\n{menu_text}\n\n请根据上面的要求输出 JSON。"

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return json.loads(resp.choices[0].message.content)


def build_competitor_profiles(
    competitors_df: pd.DataFrame,
    api_key: str,
    max_n: int = 15,
) -> List[Dict[str, Any]]:
    """
    将附近竞争对手的基础信息 + Google 详情整理成给 AI 用的简洁结构。
    为控制调用次数，只取评分靠前的前 max_n 家。
    """
    profiles: List[Dict[str, Any]] = []
    if competitors_df is None or competitors_df.empty:
        return profiles

    subset = competitors_df.head(max_n)
    for _, row in subset.iterrows():
        pid = row.get("place_id")
        if not pid:
            continue
        try:
            detail = google_place_details(api_key, pid)
        except Exception:
            detail = {}

        profiles.append(
            {
                "name": detail.get("name") or row.get("name"),
                "vicinity": detail.get("formatted_address") or row.get("vicinity"),
                "rating": detail.get("rating") or row.get("rating"),
                "reviews": detail.get("user_ratings_total") or row.get("reviews"),
                "price_level": detail.get("price_level"),
                "types": detail.get("types", []),
            }
        )
    return profiles


def rank_competitors_with_gpt(
    profile: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    让 ChatGPT 在候选餐厅中挑出真正的 5–10 家核心竞对，并按相似度排序。
    """
    if client is None:
        return []

    system_prompt = """
你是一名熟悉北美中餐市场的竞对分析师。
现在有一间目标餐厅的菜系画像，以及一批附近候选餐厅的信息。
请你从候选中选出最像的 5-10 家竞对，并按相似度从高到低排序。

相似度判断维度包括：
- 菜系 / 类别是否接近（比如都是川菜、粤菜、港式茶餐厅等）
- 价格带是否接近
- 是否属于相似业态（正餐/快餐/茶餐厅/奶茶店/烘焙店）
- 若信息有限，可根据分类 types 和餐厅名称做合理推断

输出 JSON 对象：
{
  "competitors": [
    {
      "name": "...",
      "similarity_score": 0-100,
      "main_reason": "1-2 句中文解释",
      "vicinity": "...",
      "rating": 4.5,
      "reviews": 123,
      "price_level": 2,
      "types": ["chinese", "restaurant"]
    },
    ...
  ]
}
    """.strip()

    user_content = {
        "target_profile": profile,
        "candidates": candidates,
    }

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    data = json.loads(resp.choices[0].message.content)
    return data.get("competitors", [])

# =========================
# ChatGPT 深度分析函数
# =========================

def call_llm_safe(messages: List[Dict[str, Any]]) -> str:
    if client is None:
        return "未配置 OPENAI_API_KEY，无法调用 ChatGPT，请在 Streamlit Secrets 中添加 OPENAI_API_KEY。"
    try:
        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            temperature=0.4,
        )
        return completion.choices[0].message.content
    except Exception as e:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.4,
            )
            return completion.choices[0].message.content
        except Exception as e2:
            return f"调用 ChatGPT 失败。\n主模型错误：{e}\n备用模型错误：{e2}"


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
   - 判断该店的主菜系和子菜系（例如：粤菜-茶餐厅、川菜-辣炒、东北家常菜、上海菜等），说明依据。
   - 如果菜单里有多种菜系，请说明主次结构。

2. **菜单结构与价格带分析**
   - 根据菜单文本，分析：
     - 热门品类（如主食类、招牌菜、套餐、炸鸡、甜品等）
     - 人均价位区间、主力价格带（例如：多数主菜集中在 $15–$22）
     - 是否存在明显的“利润杀手”（价格偏低但制作复杂、毛利低的菜）。

3. **线上曝光 & 竞争态势解读**
   - 结合 GBP 评分、网站得分、关键词排名结果，判断：
     - 目前在本地搜索中的位置（落后程度、有无机会冲击 Top 3）。
     - 和 3–5 家核心竞品相比的明显短板和优势。

4. **外卖平台机会点（如果菜单里出现外卖平台链接）**
   - 根据菜品结构和价格，判断适合重点发力的平台类型（聚合外卖 / 自配送 / 线下堂食引流）。
   - 给出 2–3 个具体可执行的促销活动建议（比如：高毛利品类做 BOGO、午市定价逻辑等）。

5. **接下来 30 天可执行的行动清单**
   - 用清单方式给出 5–8 条“餐馆老板能听懂、能马上执行”的改进建议：
     - Google 资料 & 网站内容优先级；
     - 菜单结构和定价优化；
     - 外卖活动 & 转化率优化建议。

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

run_btn = False

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

    if run_btn:
        st.session_state["analysis_ready"] = True
else:
    st.info("先输入地址并点击“根据地址查找附近餐厅”。")

# =========================
# 3️⃣ 主分析逻辑
# =========================

if candidate_places and selected_place_id and (
    run_btn or st.session_state.get("analysis_ready", False)
):
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

    # =============================
    # 8️⃣ Google 菜单图片 → 自动 OCR
    # =============================
    st.markdown("## 8️⃣ Google 菜单图片 → 自动 OCR 提取菜品及价格（可选）")

    menu_photos = get_place_photos(place_detail, max_photos=20)

    if not menu_photos:
        st.info("没有从 Google 图片中自动识别出菜单页，将跳过图片 OCR。")
    else:
        st.write(f"已从 Google 图片中自动识别出 {len(menu_photos)} 张可能是菜单页的图片：")
        cols = st.columns(4)
        for i, item in enumerate(menu_photos):
            with cols[i % 4]:
                st.image(item["image_bytes"], use_column_width=True)

        auto_ocr_btn = st.button("🧾 自动对菜单页做 OCR 并提取菜单文本")

        if auto_ocr_btn:
            if client is None:
                st.error("未配置 OPENAI_API_KEY，无法进行 OCR。")
            else:
                ocr_results = []
                with st.spinner("AI 正在识别菜单页中的菜名和价格…"):
                    for item in menu_photos:
                        text = ocr_menu_from_image_bytes(item["image_bytes"])
                        if text:
                            ocr_results.append(text)

                if ocr_results:
                    st.session_state["ocr_menu_texts"] = ocr_results
                    st.success(f"从菜单页图片中提取出 {len(ocr_results)} 段菜单文本。")
                    for idx, txt in enumerate(ocr_results, start=1):
                        st.markdown(f"**OCR 菜单 #{idx}：**")
                        st.code(txt, language="text")
                else:
                    st.warning("自动识别的菜单页中没有提取出有效菜单文本。")

    # =============================
    # 9️⃣ 菜单抓取（官网/外卖链接）+ 合并 OCR 菜单
    # =============================

    st.markdown("## 9️⃣ 菜单抓取 & AI 菜系 / 菜单结构分析")

    auto_menu_urls = discover_menu_urls(place_detail, website_html)
    auto_menu_urls_str = "\n".join(auto_menu_urls)

    st.markdown("#### 菜单链接抓取（可手动增删）")
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

    # 把 OCR 出来的菜单文本也塞进 menus_payload（作为额外来源）
    ocr_texts = st.session_state.get("ocr_menu_texts", [])
    for idx, txt in enumerate(ocr_texts, start=1):
        menus_payload.append(
            {
                "source": f"google_menu_photo_{idx}",
                "url": "",
                "status": "ocr_ok",
                "menu_text": txt,
            }
        )

    # ========== 基于菜单菜系画像的精准竞对模块 ==========
    st.markdown("### 🍜 基于菜单菜系画像的精准竞对（实验功能）")

    ai_comp_btn = st.button("✨ 生成菜系画像 + 精准竞对列表")

    if ai_comp_btn:
        if client is None:
            st.error("未配置 OPENAI_API_KEY，无法进行菜系画像和竞对筛选。")
        else:
            combined_menu_text_parts = [m["menu_text"] for m in menus_payload if m.get("menu_text")]
            combined_menu_text = "\n".join(combined_menu_text_parts)

            if not combined_menu_text.strip():
                st.warning("当前未能成功获取任何菜单文本，无法进行菜系画像。请检查菜单链接或菜单图片 OCR。")
            else:
                with st.spinner("AI 正在根据菜单生成菜系画像…"):
                    profile = analyze_menu_profile(combined_menu_text)

                if "error" in profile:
                    st.error(profile["error"])
                else:
                    st.subheader("🔎 AI 菜系画像")
                    st.json(profile)

                    if competitors_df is None or competitors_df.empty:
                        st.info("附近竞争对手数据不足，无法进一步筛选真正竞对。")
                    else:
                        with st.spinner("AI 正在基于菜系画像筛选真正的核心竞对…"):
                            candidate_profiles = build_competitor_profiles(
                                competitors_df, GOOGLE_API_KEY, max_n=15
                            )
                            ranked_competitors = rank_competitors_with_gpt(
                                profile, candidate_profiles
                            )

                        if not ranked_competitors:
                            st.warning("AI 未能返回有效的竞对列表，可能是信息太少或模型调用出错。")
                        else:
                            st.subheader("🏆 AI 判定的核心竞对（按相似度排序）")
                            ranked_df = pd.DataFrame(ranked_competitors)
                            st.dataframe(ranked_df, use_container_width=True)

    st.markdown("### 🧠 生成 ChatGPT 菜系 & 菜单 & 运营深度分析报告")

    ai_btn = st.button("📊 生成 AI 深度分析报告（长文版）")

    if ai_btn:
        st.info("已收到生成请求，正在调用 ChatGPT ...")

        if client is None:
            st.error("当前未配置 OPENAI_API_KEY，无法调用 ChatGPT，请在 Streamlit Secrets 中添加 OPENAI_API_KEY。")
        else:
            import traceback

            with st.spinner("正在调用 ChatGPT 生成分析报告…"):
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
                    st.error(f"调用 ChatGPT 时发生未捕获错误：{e}")
                    st.code(traceback.format_exc())

    st.markdown("## 🔟 免费获取完整诊断报告 & 1 对 1 咨询")

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

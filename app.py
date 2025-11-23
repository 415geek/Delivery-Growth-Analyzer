import streamlit as st
import pandas as pd
import numpy as np

# ===================== 基本配置 ===================== #
st.set_page_config(
    page_title="外卖增长潜力诊断器",
    layout="wide"
)

st.title("📈 餐厅外卖增长潜力诊断器（MVP 版）")
st.caption("输入餐厅基础数据，系统基于菜单结构 & 竞对状况，预估精细化运营后的外卖提升空间。")


# ===================== 一些常量配置 ===================== #

# 行业经验：精细化运营后，正常可提升 15%~60%
MIN_GROWTH = 0.15
MAX_GROWTH = 0.60


# ===================== Mock 数据层（后面替换为真实 API/爬虫） ===================== #

def mock_fetch_yelp_basic(address: str) -> dict:
    """
    模拟通过 Yelp API 获取餐厅基础信息。
    未来可替换为真实 Yelp Fusion API 调用。
    """
    return {
        "name": "Demo Bistro",
        "rating": 4.3,
        "review_count": 256,
        "price_level": "$$",
        "categories": ["Chinese", "Noodles"],
        "lat": 37.78,
        "lng": -122.41,
    }


def mock_fetch_google_dinein_menu(address: str) -> pd.DataFrame:
    """
    模拟通过 Google Places API 获取堂食菜单。
    返回 DataFrame: name, price, category, channel
    """
    data = [
        {"name": "Spicy Beef Noodle", "price": 15.5, "category": "Noodles", "channel": "dine-in"},
        {"name": "Wonton Soup", "price": 9.9, "category": "Appetizer", "channel": "dine-in"},
        {"name": "Fried Rice", "price": 14.0, "category": "Rice", "channel": "dine-in"},
        {"name": "Coke", "price": 3.5, "category": "Drinks", "channel": "dine-in"},
    ]
    return pd.DataFrame(data)


def mock_fetch_doordash_menu(address: str) -> pd.DataFrame:
    """
    模拟 Doordash 外卖菜单。
    未来可以用 requests + BeautifulSoup 解析店铺页面 HTML。
    """
    data = [
        {"name": "Spicy Beef Noodle", "price": 18.5, "category": "Noodles", "channel": "doordash", "tags": ["popular"]},
        {"name": "Wonton Soup", "price": 11.5, "category": "Appetizer", "channel": "doordash", "tags": []},
        {"name": "Fried Rice Combo", "price": 19.9, "category": "Combo", "channel": "doordash", "tags": ["most loved"]},
        {"name": "Orange Chicken", "price": 17.9, "category": "Entrees", "channel": "doordash", "tags": []},
        {"name": "Coke", "price": 4.5, "category": "Drinks", "channel": "doordash", "tags": []},
    ]
    return pd.DataFrame(data)


def mock_fetch_ubereats_menu(address: str) -> pd.DataFrame:
    """
    模拟 Uber Eats 外卖菜单。
    """
    data = [
        {"name": "Spicy Beef Noodle", "price": 17.9, "category": "Noodles", "channel": "ubereats", "tags": ["top ordered"]},
        {"name": "Wonton Soup", "price": 10.9, "category": "Appetizer", "channel": "ubereats", "tags": []},
        {"name": "Fried Rice", "price": 16.8, "category": "Rice", "channel": "ubereats", "tags": []},
        {"name": "Orange Chicken", "price": 18.5, "category": "Entrees", "channel": "ubereats", "tags": ["popular"]},
    ]
    return pd.DataFrame(data)


def mock_fetch_competitors(lat: float, lng: float) -> pd.DataFrame:
    """
    模拟附近 1km 竞对信息（正常应来自 Yelp + DD + UE 检索）。
    """
    data = [
        {"name": "Nearby Noodle House", "rating": 4.6, "price_level": "$$", "distance_km": 0.3},
        {"name": "Spicy Hotpot & Rice", "rating": 4.1, "price_level": "$$", "distance_km": 0.5},
        {"name": "Healthy Bowl & Salad", "rating": 4.7, "price_level": "$$", "distance_km": 0.8},
    ]
    return pd.DataFrame(data)


# ===================== 分析逻辑层 ===================== #

def compute_menu_structure_score(df_all: pd.DataFrame) -> (float, list):
    """
    维度1：菜单结构健康度（0–100）
    简单规则版：越极端越扣分。
    """
    tips = []

    if df_all.empty:
        return 50.0, ["未获取到菜单数据，使用默认中性评分。"]

    total_items = len(df_all)
    num_categories = df_all["category"].nunique()

    score = 100.0

    # 单品太少 / 太多
    if total_items < 10:
        score -= 15
        tips.append("外卖菜单单品过少，用户选择有限，建议补充 2–3 个高毛利 Star Item。")
    elif total_items > 60:
        score -= 25
        tips.append("外卖菜单单品超过 60 个，容易导致选择困难，建议精简和合并部分菜品。")

    # 类目太多
    if num_categories > 8:
        score -= 15
        tips.append("菜单类别过多，建议压缩到 5–7 个主类目，突出主力品类。")

    # 判断是否有组合餐 (Combo)
    if "Combo" not in [c.lower() for c in df_all["category"].unique()]:
        score -= 10
        tips.append("缺少套餐/组合菜单，建议设计 2–3 个客单价更高的套餐组合，提升客单价。")

    return max(score, 0), tips


def compute_pricing_score(df_dinein: pd.DataFrame, df_delivery: pd.DataFrame) -> (float, list):
    """
    维度2：定价与客单价策略（0–100）
    对比堂食 vs 外卖价格加价率，是否合理。
    """
    tips = []
    if df_dinein.empty or df_delivery.empty:
        return 60.0, ["缺少堂食或外卖价格数据，暂用中性评分。"]

    # 按菜名 merge（真实生产环境要做模糊匹配，这里简单处理）
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


def compute_promotion_score() -> (float, list):
    """
    维度3：活动体系（0–100）
    当下用 mock：假设商家活动较弱。
    未来可根据 DD/UE 页面解析 “$X off”、“BOGO” 等。
    """
    score = 55.0
    tips = [
        "当前在外卖平台上的优惠活动较弱或不连续，建议设计：首单减免、老客满减、午晚高峰分时段优惠等精细化活动矩阵。"
    ]
    return score, tips


def compute_competitor_score(df_comp: pd.DataFrame, restaurant_rating: float) -> (float, list):
    """
    维度4：竞对压力指数（0–100）
    看自己评分 vs 附近评分均值。
    """
    tips = []
    if df_comp.empty:
        return 60.0, ["未获取到竞对数据，暂用中性评分。"]

    avg_comp_rating = df_comp["rating"].mean()
    diff = restaurant_rating - avg_comp_rating

    score = 60.0 + diff * 10  # 每高 0.1 分加 1 分
    score = max(min(score, 100.0), 0.0)

    if diff >= 0.2:
        tips.append(f"本店评分 {restaurant_rating:.1f} 高于附近竞对均值 {avg_comp_rating:.1f}，口碑具备优势，可以在外卖详情页更突出。")
    elif diff <= -0.2:
        tips.append(f"本店评分 {restaurant_rating:.1f} 低于附近竞对均值 {avg_comp_rating:.1f}，建议通过服务、包装、好评激励活动快速拉升评分。")
    else:
        tips.append("本店评分与附近竞对大致持平，建议通过菜品照片与活动玩法做差异化。")

    return score, tips


def compute_coverage_score() -> (float, list):
    """
    维度5：配送覆盖 & 热区（0–100）
    当前用 mock：给一个中性偏乐观分数。
    未来可结合平台配送半径 + 周边人群热力。
    """
    score = 70.0
    tips = [
        "从地理位置和商圈结构看，配送覆盖具备一定潜力，后续可结合配送半径与学校/写字楼密度做更精细评估。"
    ]
    return score, tips


def compute_growth_rate(menu_score, price_score, promo_score, comp_score, coverage_score) -> float:
    """
    汇总五大维度，计算“潜在增长率”（0~1）。
    再夹在 MIN_GROWTH ~ MAX_GROWTH 之间。
    """
    weighted = (
        0.20 * menu_score +
        0.15 * price_score +
        0.25 * promo_score +
        0.15 * comp_score +
        0.25 * coverage_score
    ) / 100.0  # 转成 0~1

    # 把线性结果映射到 [MIN_GROWTH, MAX_GROWTH]
    growth_rate = MIN_GROWTH + (MAX_GROWTH - MIN_GROWTH) * weighted
    return growth_rate


# ===================== 主分析函数 ===================== #

def analyze_restaurant(address: str, avg_orders: float, avg_ticket: float) -> dict:
    """
    核心分析管线：
    1. 获取 Yelp 基础信息
    2. 获取堂食 & 外卖菜单
    3. 获取竞对信息
    4. 计算五大评分
    5. 估算外卖提升空间
    """
    # 1. 基础信息 & 竞对
    yelp_info = mock_fetch_yelp_basic(address)
    comp_df = mock_fetch_competitors(yelp_info["lat"], yelp_info["lng"])

    # 2. 菜单数据
    dinein_df = mock_fetch_google_dinein_menu(address)
    dd_df = mock_fetch_doordash_menu(address)
    ue_df = mock_fetch_ubereats_menu(address)

    all_df = pd.concat([dinein_df, dd_df, ue_df], ignore_index=True)

    # 3. 各维度评分
    menu_score, menu_tips = compute_menu_structure_score(all_df)
    price_score, price_tips = compute_pricing_score(dinein_df, dd_df)
    promo_score, promo_tips = compute_promotion_score()
    comp_score, comp_tips = compute_competitor_score(comp_df, yelp_info["rating"])
    coverage_score, coverage_tips = compute_coverage_score()

    # 4. 潜在增长率 & 营业额提升
    growth_rate = compute_growth_rate(menu_score, price_score, promo_score, comp_score, coverage_score)

    current_daily_revenue = avg_orders * avg_ticket
    potential_daily_revenue = current_daily_revenue * (1 + growth_rate)
    revenue_uplift_daily = potential_daily_revenue - current_daily_revenue
    revenue_uplift_monthly = revenue_uplift_daily * 30

    result = {
        "yelp_info": yelp_info,
        "competitors": comp_df,
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
        with st.spinner("正在基于菜单 & 竞对数据进行诊断，请稍等..."):
            result = analyze_restaurant(address, avg_orders, avg_ticket)

        # 顶部 KPI 区
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
            f"基于当前菜单结构、定价策略、活动体系、竞对压力与覆盖情况，"
            f"系统预估通过精细化运营，可带来约 **{result['growth_rate']*100:.1f}%** 的外卖营业额增长空间。"
        )

        # 评分雷达 / 柱状展示
        st.subheader("🧬 五大维度诊断评分")
        score_df = pd.DataFrame(
            {
                "维度": list(result["scores"].keys()),
                "得分": list(result["scores"].values()),
            }
        )
        st.bar_chart(score_df.set_index("维度"))

        # 分维度建议
        st.subheader("🩺 分维度运营建议（可以直接和老板讲人话）")
        for dim, tips in result["tips"].items():
            with st.expander(f"{dim} · 诊断与建议"):
                for t in tips:
                    st.markdown(f"- {t}")

        # 菜单对比展示
        st.subheader("📑 堂食 vs 外卖菜单结构对比（Demo 数据）")

        tab1, tab2, tab3, tab4 = st.tabs(["堂食菜单", "Doordash 菜单", "UberEats 菜单", "整合视图"])
        with tab1:
            st.dataframe(result["menus"]["dinein"])
        with tab2:
            st.dataframe(result["menus"]["doordash"])
        with tab3:
            st.dataframe(result["menus"]["ubereats"])
        with tab4:
            st.dataframe(result["menus"]["all"])

        # 竞对概览
        st.subheader("🏁 附近竞对概览（Demo 数据）")
        st.dataframe(result["competitors"])

        st.info(
            "当前版本为 MVP Demo：外卖菜单 & 竞对数据使用的是示例数据结构。"
            "后续可以逐步接入 Yelp / Google 官方 API，以及 Doordash / Uber Eats 页面解析，实现真实线上诊断。"
        )
else:
    st.markdown(
        """
        ### 使用说明（MVP 思路）
        1. 输入餐厅地址 + 当前日均外卖单量 + 客单价  
        2. 系统会：
           - 获取餐厅基础信息 & 附近竞对（当前为 Demo 数据）  
           - 整合堂食与外卖菜单  
           - 基于五大维度打分：菜单结构 / 定价策略 / 活动体系 / 竞对压力 / 覆盖圈层  
           - 计算预计可提升的外卖营业额（按天 & 按月）  
        3. 输出一份可以给老板看得懂、你自己拿得出手的「增长诊断报告」。  
        """
    )

# demo py
import streamlit as st
import pandas as pd
import plotly.express as px
from analyze import analyze
from html import escape

# ========页面标题=========
st.set_page_config(page_title="校园诉求小帮手", page_icon=":school:", layout="wide")
st.markdown(
    """
    <style>
    /* 限制内容宽度，避免大屏幕上拉得太宽 */
    .stMainBlockContainer {
        max-width: 1000px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* 带边框的 Streamlit 容器：变成白色卡片 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border-radius: 18px;
        box-shadow: 0 6px 20px rgba(23, 43, 77, 0.05);
    }

    /* 只调整主要按钮 */
    button[kind="primary"] {
        border-radius: 12px;
        min-height: 48px;
        font-weight: 600;
    }

    /* 分析结果卡片 */
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #E1E9F5;
        border-radius: 16px;
        padding: 24px 16px;
        min-height: 175px;
        text-align: center;
        box-sizing: border-box;
        box-shadow: 0 6px 20px rgba(23, 43, 77, 0.05);
    }

    .result-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 52px;
        height: 52px;
        border-radius: 50%;
        font-size: 26px;
        margin-bottom: 12px;
    }

    .result-label {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .result-value {
        color: #172B4D;
        font-size: 22px;
        font-weight: 700;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("校园诉求小帮手")

# ========标签页:学生端/管理员端=========
tab_student, tab_admin = st.tabs(["学生端（提交诉求）", "管理员端（统计数据）"])


# ------------ 学生端
def show_result_card(icon, title, value, background, color):
    safe_value = escape(str(value))

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-icon"
                 style="background-color: {background}; color: {color};">
                {icon}
            </div>
            <div class="result-label">{title}</div>
            <div class="result-value">{safe_value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with tab_student:
    with st.container(border=True):
        st.subheader("提交你的诉求")
        st.caption("请描述具体情况，不超过 200 字。")

        content = st.text_area(
            "诉求内容",
            placeholder="例如：宿舍洗衣机坏了三天，希望尽快维修。",
            max_chars=200,
            height=160,
        )

        submit_btn = st.button(
            "提交并分析",
            type="primary",
            width="stretch",
        )
    if submit_btn:
        if not content.strip():
            st.warning("请输入诉求内容后再提交。")
        else:
            try:
                result = analyze(content)
                summary = result["summary"]
                label = result["label"]
                urgency = result["urgency"]
                department = result["department"]
                st.subheader("分析结果")

                urgency_colors = {
                    "低": ("#E8F7EE", "#15803D"),
                    "中": ("#FFF4D6", "#B7791F"),
                    "高": ("#FEECEC", "#DC2626"),
                }

                urgency_background, urgency_color = urgency_colors.get(
                    urgency,
                    ("#EEF2F7", "#64748B"),
                )

                col_summary, col_category, col_urgency, col_department = st.columns(4)

                with col_summary:
                    show_result_card(
                        "🍣",
                        "摘要",
                        summary,
                        "#E8F0FF",
                        "#EB2546",
                    )

                with col_category:
                    show_result_card(
                        "🏠",
                        "类别",
                        label,
                        "#E8F0FF",
                        "#2563EB",
                    )

                with col_urgency:
                    show_result_card(
                        "!",
                        "紧急度",
                        urgency,
                        urgency_background,
                        urgency_color,
                    )

                with col_department:
                    show_result_card(
                        "👥",
                        "负责部门",
                        department,
                        "#E8F7EE",
                        "#15803D",
                    )
                if result["error"] is not None:
                    st.error(f"错误：{result['error']}")

                # 追加写入csv
                new_row = pd.DataFrame({"summary": [summary], "label": [label], "urgency": [urgency], "department": [department]})
                new_row.to_csv("data/history.csv", mode="a", header=False, index=False, encoding="utf-8-sig")

            except Exception as e:
                st.error(f"❌ 分析失败: {e}")


# ------------ 管理员端
with tab_admin:
    st.header("管理端首页 - 诉求统计数据面板")
    st.divider()

    # 读取csv,增加异常保护,防止空文件报错崩溃
    try:
        df = pd.read_csv("data/history.csv")
        st.success("✅ 数据加载成功")
    except pd.errors.EmptyDataError:
        st.warning("history.csv文件为空!")
        df = pd.DataFrame(columns=["content", "label", "urgency", "department"])

    # 基础指标
    total_count = len(df)
    high_urgency_count = len(df[df["urgency"] == "高"])
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="总诉求数量", value=total_count)
    with col2:
        st.metric(label="高紧急度诉求数量", value=high_urgency_count)

    st.divider()

    # 两张可视化图表
    if not df.empty:
        st.subheader("诉求分类统计")
        cat_count = df["label"].value_counts().reset_index()
        fig1 = px.bar(cat_count, x="label", y="count", title="各类诉求数量")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("紧急程度分布")
        urg_count = df["urgency"].value_counts().reset_index()
        fig2 = px.pie(urg_count, names="urgency", values="count", title="紧急程度分布")
        st.plotly_chart(fig2)

        st.divider()
        st.subheader("原始诉求数据表")
        st.dataframe(df)

    else:
        st.info("暂无诉求数据,提交诉求后图标会自动生成")

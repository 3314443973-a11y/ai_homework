# demo py
import streamlit as st
import pandas as pd
import plotly.express as px
from analyze import analyze

# ========页面标题=========
st.set_page_config(page_title="校园诉求小帮手", page_icon=":school:", layout="wide")
st.title("校园诉求小帮手")

# ========标签页:学生端/管理员端=========
tab_student, tab_admin = st.tabs(["学生端（提交诉求）", "管理员端（统计数据）"])


# ------------ 学生端
with tab_student:
    st.header("学生诉求提交")
    st.write("请输入您的诉求内容,不超过200字")
    content = st.text_area("诉求内容", max_chars=200, label_visibility="collapsed")
    submit_btn = st.button("提交并分析")

    if submit_btn:
        if not content.strip():
            st.warning("请输入诉求内容后再提交。")
        else:
            try:
                result = analyze(content)
                label = result["label"]
                urgency = result["urgency"]
                department = result["department"]
                st.write("分析结果")
                st.write(f"类别：{result['label']}")
                st.write(f"紧急度：{result['urgency']}")
                st.write(f"负责部门：{result['department']}")
                if result["error"] is not None:
                    st.error(f"错误：{result['error']}")

                # 追加写入csv
                new_row = pd.DataFrame({"content": [content], "label": [label], "urgency": [urgency], "department负责部门": [department]})
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

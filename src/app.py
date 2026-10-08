# demo py
import streamlit as st


def analyze_text(input_text):
    result = {"类别": "示例类别", "紧急程度": "示例紧急程度", "处理部门": "示例处理部门"}
    return result


st.title("文本分析工具")
text = st.text_area("请输入文本进行分析", height=150)
if st.button("提交"):
    if text.strip() == "":
        st.warning("请输入文本后再提交")
    else:
        st.write(analyze_text(text))

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


def predict(text):
    # 空输入检验
    if not text.strip():
        return "请输入文本后再提交"

    # 超长文本拦截200字
    max_len = 200
    if len(text) > max_len:
        text = text[:max_len]

    try:
        x = vectorizer.transform([text])
        pred = model.predict(x)
        return {"code": 0, "result": str(pred), "notice": "结果仅供参考，请结合实际情况判断"}

    except Exception as e:
        return {"code": 1, "result": str(e), "notice": "预测过程中出现错误，请检查输入文本或联系管理员"}

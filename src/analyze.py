"""
统一分析入口
输入：学生诉求字符串
输出字典:「分类label,紧急程度urgency,处理部门department」
"""

from predict import predict_label
from urgency import get_urgency
from department import get_department
from summary_2 import summarize


def analyze(text: str) -> dict:
    # 1.调用A的接口分析文本
    label = predict_label(text)
    # 2.调用B的接口分析紧急度
    urgency = get_urgency(text)
    # 3.调用B的接口分析部门
    department = get_department(label)
    # 4.调用D的接口分析摘要
    info = summarize(text)

    # 将提取结果组合成文字
    parts = [
        info["location"],
        info["problem"],
        info["time"],
    ]

    if info["request"]:
        parts.append("希望" + info["request"])

    # 过滤空字段；全部为空时显示原文
    summary = ",".join(part for part in parts if part) or text.strip()

    return {
        "summary": summary,
        "label": label,
        "urgency": urgency,
        "department": department,
        "error": None,
    }

"""
统一分析入口
输入：学生诉求字符串
输出字典:「分类label,紧急程度urgency,处理部门department」
"""

from predict import predict_label
from urgency import get_urgency
from department import get_department


def analyze(text: str) -> dict:
    # 1.调用A的接口分析文本
    label = predict_label(text)
    # 2.调用B的接口分析紧急度
    urgency = get_urgency(text)
    # 3.调用B的接口分析部门
    department = get_department(label)

    return {
        "label": label,
        "urgency": urgency,
        "department": department,
        "error": None,
    }

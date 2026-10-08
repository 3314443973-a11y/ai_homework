"""Map the final complaint label to its responsible department."""

LABEL_TO_DEPARTMENT = {
    "宿舍设施": "慧湖物业中心",
    "校园网络": "网络管理部",
    "食堂餐饮": "餐饮管理",
    "教学设施": "教学保障",
    "校园安全": "保卫处",
    "其他": "人工处理",
}


def get_department(label: str) -> str:
    """Return the department for a validated label.

    The public interface intentionally accepts ``label`` only. Unknown labels
    fail loudly so an upstream classification or schema error is not hidden.
    """
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label不能为空")
    normalized = label.strip()
    try:
        return LABEL_TO_DEPARTMENT[normalized]
    except KeyError as exc:
        allowed = "、".join(LABEL_TO_DEPARTMENT)
        raise ValueError(f"未知label：{normalized}；允许值：{allowed}") from exc

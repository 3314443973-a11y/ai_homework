"""Explainable urgency rules for campus complaints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UrgencyResult:
    urgency: str
    reason: str
    matched_rule: str
    needs_review: bool = False


NEGATIONS = ("没有", "没", "未", "并无", "不是", "不再", "不存在", "否认")

# Direct threats to student/staff safety or health.
HIGH_RISK_TERMS = (
    "明火", "着火", "起火", "火灾", "冒烟", "火花",
    "漏电", "触电", "电线裸露", "裸露电线", "电源线破损",
    "持刀", "持棍", "暴力威胁", "打架", "袭击",
    "食物中毒", "食品中毒", "集体呕吐", "多人呕吐", "多人腹泻",
    "严重过敏", "呼吸困难", "昏迷", "晕倒", "大量出血",
    "安全出口堵塞", "消防通道堵塞", "消防通道上锁",
    "井盖破损", "护栏松动", "扶手松动", "差点摔倒", "已经摔倒",
    "随时可能掉落", "即将坠落", "有坠落风险", "差点撞到行人",
)

# A lack of essential protection can be a health threat even without an
# electrical/fire keyword.
HEALTH_PROTECTION_PATTERNS = (
    ("被子", "没有被子"),
    ("被子", "没被子"),
    ("被子", "无被子"),
    ("被子", "被锁在外面"),
    ("停水", "整个宿舍"),
    ("停水", "整个园区"),
    ("消防通道", "上锁"),
    ("疏散出口", "堵"),
    ("门锁", "进不了房间"),
    ("反锁", "进不去"),
    ("多人", "呕吐", "腹泻"),
    ("多名", "呕吐", "腹泻"),
    ("施工区域", "缺少", "夜间"),
    ("逆行", "差点撞到"),
)

CRITICAL_OPERATION_PATTERNS = (
    ("全校", "网络", "中断"),
    ("多个教学楼", "网络", "中断"),
    ("线上考试", "停"),
    ("最后一天", "全体", "空白文件"),
)

# Services with a short operating window whose failure can directly affect
# later study arrangements.
TIME_CRITICAL_SERVICE_CONTEXTS = ("选课",)

# Any reported physical symptom after canteen food consumption is treated as
# a food-safety risk, regardless of how many people are affected.
CANTEEN_FOOD_CONTEXTS = (
    "食堂", "餐厅", "饭菜", "餐食", "食品", "菜品",
    "就餐", "用餐", "吃完",
)
CANTEEN_HEALTH_SYMPTOMS = (
    "腹痛", "肚子痛", "呕吐", "腹泻", "拉肚子",
    "恶心", "头晕", "发热", "发烧", "过敏反应", "出现过敏",
    "发生过敏", "身体不适",
)

LARGE_SCALE_TERMS = (
    "全校", "整栋", "整层", "整个宿舍园区", "多个教学楼", "大面积",
)

SERVICE_FAILURE_TERMS = (
    "无法", "不能", "连不上", "打不开", "不制冷", "不出水", "停水",
    "停电", "断网", "故障", "损坏", "坏了", "堵塞", "失灵", "无法使用",
    "认证失败", "重复扣费", "一直漏水", "渗水", "中断", "加载失败",
)

MEDIUM_IMPACT_TERMS = (
    "排队太久", "一直没有空位", "无法预约", "持续没有空位", "严重影响",
)

LOW_INTENT_TERMS = (
    "建议", "希望增加", "希望延长", "希望优化", "希望改进", "可以增加",
    "能不能增加", "体验", "选择太少", "偏咸", "声音很大", "速度比较慢",
    "偶尔", "目前还能", "仍能使用", "还能使用", "不影响使用",
    "逐步更换", "标识不清楚", "指示不够清楚", "排队的人太多",
    "出水速度异常", "仍没有改善",
)


def _is_negated(text: str, term: str, window: int = 6) -> bool:
    """Return whether a matched risk term is negated immediately before it."""
    start = 0
    found_negated_occurrence = False
    while True:
        index = text.find(term, start)
        if index < 0:
            return found_negated_occurrence
        prefix = text[max(0, index - window):index]
        if any(negation in prefix for negation in NEGATIONS):
            found_negated_occurrence = True
            start = index + len(term)
            continue
        return False


def _first_active_term(text: str, terms: tuple[str, ...]) -> str | None:
    for term in terms:
        if term in text and not _is_negated(text, term):
            return term
    return None


def analyze_urgency(text: str) -> UrgencyResult:
    """Classify urgency as high, medium, or low with an explanation."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text不能为空")
    normalized = "".join(text.strip().split())

    for required_terms in HEALTH_PROTECTION_PATTERNS:
        if all(term in normalized for term in required_terms):
            return UrgencyResult(
                urgency="高",
                reason="缺少基本生活保护条件，存在明确健康风险",
                matched_rule="health_protection",
            )

    for required_terms in CRITICAL_OPERATION_PATTERNS:
        if all(term in normalized for term in required_terms):
            return UrgencyResult(
                urgency="高",
                reason="关键业务或明确截止期限受到大范围影响，需要立即响应",
                matched_rule="critical_operation_failure",
            )

    canteen_symptom = _first_active_term(normalized, CANTEEN_HEALTH_SYMPTOMS)
    if canteen_symptom and any(term in normalized for term in CANTEEN_FOOD_CONTEXTS):
        return UrgencyResult(
            urgency="高",
            reason=f"餐饮或食品场景中出现身体症状：{canteen_symptom}",
            matched_rule="canteen_food_safety_health_risk",
        )

    high_term = _first_active_term(normalized, HIGH_RISK_TERMS)
    if high_term:
        return UrgencyResult(
            urgency="高",
            reason=f"检测到现实安全或健康威胁：{high_term}",
            matched_rule="direct_safety_or_health_threat",
        )

    service_term = _first_active_term(normalized, SERVICE_FAILURE_TERMS)
    if service_term and any(term in normalized for term in TIME_CRITICAL_SERVICE_CONTEXTS):
        return UrgencyResult(
            urgency="高",
            reason=f"有明确办理时限的关键学习业务发生故障：{service_term}",
            matched_rule="time_critical_service_failure",
        )

    if service_term and any(term in normalized for term in LARGE_SCALE_TERMS):
        return UrgencyResult(
            urgency="高",
            reason=f"大范围服务中断：{service_term}",
            matched_rule="large_scale_failure",
        )

    if service_term:
        return UrgencyResult(
            urgency="中",
            reason=f"影响正常学习生活：{service_term}",
            matched_rule="service_failure",
        )

    medium_term = next((term for term in MEDIUM_IMPACT_TERMS if term in normalized), None)
    if medium_term:
        return UrgencyResult(
            urgency="中",
            reason=f"持续影响正常服务获取：{medium_term}",
            matched_rule="continued_service_impact",
        )

    if any(term in normalized for term in LOW_INTENT_TERMS):
        return UrgencyResult(
            urgency="低",
            reason="一般建议或体验优化，未发现直接安全健康风险",
            matched_rule="suggestion_or_minor_issue",
        )

    return UrgencyResult(
        urgency="中",
        reason="文本未命中明确高风险或低风险规则，按一般影响处理并建议复核",
        matched_rule="default_medium",
        needs_review=True,
    )


def get_urgency(text: str) -> str:
    """Return only the public high/medium/low value."""
    return analyze_urgency(text).urgency

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from department import get_department  # noqa: E402
from urgency import analyze_urgency, get_urgency  # noqa: E402


class DepartmentTests(unittest.TestCase):
    def test_all_label_mappings(self):
        expected = {
            "宿舍设施": "慧湖物业中心",
            "校园网络": "网络管理部",
            "食堂餐饮": "餐饮管理",
            "教学设施": "教学保障",
            "校园安全": "保卫处",
            "其他": "人工处理",
        }
        for label, department in expected.items():
            with self.subTest(label=label):
                self.assertEqual(get_department(label), department)

    def test_unknown_label_fails(self):
        with self.assertRaises(ValueError):
            get_department("category")


class UrgencyTests(unittest.TestCase):
    def test_direct_fire_risk_is_high(self):
        self.assertEqual(get_urgency("教室插座正在冒烟"), "高")

    def test_blanket_locked_out_is_high(self):
        text = "被子被锁在外面，晚上没有被子盖"
        result = analyze_urgency(text)
        self.assertEqual(result.urgency, "高")
        self.assertEqual(result.matched_rule, "health_protection")

    def test_negated_electrical_risk_is_not_high(self):
        text = "宿舍插座没有漏电，只是外面的盖板松动了"
        self.assertEqual(get_urgency(text), "中")

    def test_large_scale_network_failure_is_high(self):
        self.assertEqual(get_urgency("多个教学楼同时断网，线上考试无法继续"), "高")

    def test_locked_fire_exit_is_high(self):
        self.assertEqual(get_urgency("消防通道的门被上锁，紧急时无法通过"), "高")

    def test_dorm_lockout_is_high(self):
        self.assertEqual(get_urgency("寝室门锁坏了，我们现在进不了房间"), "高")

    def test_multiple_students_sick_is_high(self):
        self.assertEqual(get_urgency("多人就餐后出现呕吐和腹泻"), "高")

    def test_multiple_named_students_sick_is_high(self):
        self.assertEqual(get_urgency("多名同学就餐后出现呕吐和腹泻"), "高")

    def test_single_person_canteen_symptom_is_high(self):
        text = "我在食堂吃完饭后出现腹痛"
        result = analyze_urgency(text)
        self.assertEqual(result.urgency, "高")
        self.assertEqual(result.matched_rule, "canteen_food_safety_health_risk")

    def test_several_students_canteen_symptoms_are_high(self):
        text = "食堂饭菜吃完后，几位同学出现腹痛和呕吐"
        self.assertEqual(get_urgency(text), "高")

    def test_negated_canteen_symptom_is_not_high(self):
        text = "食堂饭菜吃完没有腹痛，只是味道偏咸"
        self.assertEqual(get_urgency(text), "低")

    def test_allergen_labeling_request_is_not_a_reported_symptom(self):
        text = "希望食堂能标出菜品里的花生等过敏原"
        self.assertEqual(get_urgency(text), "中")

    def test_deadline_critical_failure_is_high(self):
        text = "今天是最后一天，全体毕业生的证明都生成成空白文件"
        self.assertEqual(get_urgency(text), "高")

    def test_course_selection_failure_is_high(self):
        self.assertEqual(get_urgency("教务系统选课时一直显示加载失败"), "高")

    def test_normal_service_failure_is_medium(self):
        self.assertEqual(get_urgency("宿舍宽带从昨晚开始连不上"), "中")

    def test_suggestion_is_low(self):
        self.assertEqual(get_urgency("建议增加校园文化活动的种类"), "低")

    def test_minor_signage_issue_is_low(self):
        self.assertEqual(get_urgency("分类垃圾桶标识不清楚"), "低")

    def test_persistent_unavailable_appointment_is_medium(self):
        self.assertEqual(get_urgency("心理咨询预约一直没有空位，希望增加时段"), "中")

    def test_injured_animal_does_not_trigger_student_health_rule(self):
        result = analyze_urgency("校园里有一只受伤的小猫需要帮助")
        self.assertNotEqual(result.urgency, "高")

    def test_unknown_case_requests_review(self):
        result = analyze_urgency("老师今天迟到了")
        self.assertEqual(result.urgency, "中")
        self.assertTrue(result.needs_review)


if __name__ == "__main__":
    unittest.main()

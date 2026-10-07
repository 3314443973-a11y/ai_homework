# 第 6 天规则模块交付

## 文件

- `src/urgency.py`：紧急度规则，公开函数为 `get_urgency(text)`。
- `src/department.py`：部门映射，公开函数为 `get_department(label)`。
- `tests/test_rules.py`：边界案例测试。

## 接入方式

```python
from department import get_department
from urgency import get_urgency

label = classifier.predict([text])[0]
urgency = get_urgency(text)
department = get_department(label)

result = {
    "label": label,
    "urgency": urgency,
    "department": department,
    "summary": summary,
}
```

接口只使用 `label`，不输出 `category`。

## 规则顺序

1. 缺少基本生活保护条件，例如被子被锁在外面且晚上没有被子盖，判高。
2. 火情、漏电、暴力、严重健康风险等现实威胁，判高。
3. 食堂、餐食或食品场景中，只要有人出现腹痛、呕吐、腹泻等身体症状，不论人数多少，判高。
4. 选课等有明确办理时限、影响后续学习安排的关键业务发生故障，判高。
5. 大范围服务中断，判高。
6. 一般设施或服务无法使用，判中。
7. 建议和轻微体验问题，判低。
8. 无法明确命中的文本默认判中，并设置 `needs_review=True` 供内部调试或人工复核。

否定表达会优先处理，例如“没有漏电，只是盖板松动”不会因出现“漏电”直接判高。
“希望标注花生等过敏原”属于预防建议，不代表已经有人出现过敏反应，因此不会仅因“过敏原”三个字判高。

## 边界说明

规则模块负责紧急度，不负责修改模型输出的 `label`。同时涉及多个类别时，类别仍由分类模型及最终业务规则决定。部门只由最终 `label` 映射产生。

## 评估结果

评估日期：2026-10-06。使用 `data/test_50.csv` 的 50 条数据；根据成员 B 的人工确认，将“实验室部分电脑无法正常启动实验软件”的紧急度由高修正为中，其余样本未改动。

- Accuracy：100.00%
- Macro F1：100.00%
- 高紧急度：Precision、Recall、F1 均为 100.00%
- 中紧急度：Precision、Recall、F1 均为 100.00%
- 低紧急度：Precision、Recall、F1 均为 100.00%
- 混淆矩阵（行是真实值，列是预测值，顺序为高、中、低）：`[[13, 0, 0], [0, 27, 0], [0, 0, 10]]`

评估文件位于 `results/urgency_rules/`：

- `summary.json`：完整指标。
- `confusion_matrix.csv`：混淆矩阵。
- `predictions.csv`：逐条预测、命中规则和原因。
- `mismatches.csv`：仅保留误判样本。

## 人工确认与限制

成员 B 已确认：实验室部分电脑无法启动实验软件判中；教务系统选课持续加载失败因有紧迫办理时限且影响后续学习，判高。规则已将“选课 + 实际服务故障”统一视为关键时限业务故障，而不是只记忆某一句测试文本。

`test_50.csv` 已同步修正上述实验室样本，因此当前 `mismatches.csv` 没有误判记录。

由于规则和测试标签曾根据测试样本的人工复核进行调整，当前 100% 指标应视为规则回归检查结果，不应宣称为完全独立的泛化能力。后续应使用新增、未参与规则制定的诉求进行盲测。

规则判断是可解释基线，不替代人工复核。`needs_review=True` 表示没有命中明确规则；前端正式返回仍只需使用 `urgency`，调试时可额外查看 `reason` 和 `matched_rule`。

## 交接清单

交给成员 A、C：

- 固定数据：`data/train_250.csv`、`data/test_50.csv`，不得重新划分。
- 最终分类器配置：字级 TF-IDF 1-3 gram，`min_df=2`，`sublinear_tf=True`。开发评估 Accuracy 78%，Macro F1 76.55%。
- 业务接口：`get_urgency(text)`、`get_department(label)`。
- 统一字段：`text,label,urgency,department`，禁止重新使用 `category`。

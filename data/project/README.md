# 成员 B 向 A、C 的最终交接说明

更新日期：2026-10-06。

## 运行环境

- 所有可执行程序均为 Python `.py` 文件。
- 指定 Python 版本：3.11.16。
- 已在 Python 3.11.16 环境中重新训练、测试和加载模型。
- 安装依赖：`python -m pip install -r requirements.txt`

安装后先执行环境检查：

```text
python src/verify_environment.py
```

只有输出 `"status": "ok"` 才表示Python版本、依赖和模型文件全部正确。

## 最终分类方案

- 特征：字级 TF-IDF 2-3 gram
- `analyzer="char"`
- `ngram_range=(2, 3)`
- `min_df=2`
- `sublinear_tf=True`
- 分类器：Logistic Regression
- 固定随机状态：20261002

固定测试结果：Accuracy 78%，Macro F1 77.55%，误判11条。该方案在四组实验中排名第一。

## 数据使用要求

- `data/train_250.csv`：只用于训练。
- `data/test_50.csv`：只用于训练完成后的预测和评估，不得传入 `fit()`。
- 四列固定为：`text,label,urgency,department`。
- 接口和数据只使用 `label`，不得改回 `category`。

## 直接运行

在压缩包根目录执行：

```text
python src/build_final_model.py
```

脚本会生成：

- `model/complaint_label_classifier_char_2_3.joblib`
- `model/model_summary.json`

加载模型并预测：

```python
import joblib

model = joblib.load("model/complaint_label_classifier_char_2_3.joblib")
label = model.predict([text])[0]
```

该模型是诉求类别分类器，只预测六类 `label`，不是部门分类器。部门由最终 `label` 通过固定映射产生。紧急度和部门分别调用：

```python
from src.urgency import get_urgency
from src.department import get_department

urgency = get_urgency(text)
department = get_department(label)
```

## 最终返回结构

```python
{
    "label": label,
    "urgency": urgency,
    "department": department,
    "summary": summary,
}
```

## 目录说明

- `data/`：固定训练集和测试集。
- `src/`：最终模型构建脚本、完整对照实验脚本、紧急度规则和部门映射。
- `model/`：由Python 3.11.16构建的2-3 gram诉求类别模型及指标摘要。
- `results/tfidf_jieba/`：四组TF-IDF对照结果、混淆矩阵和误判详情。
- `results/urgency_rules/`：紧急度规则回归结果。
- `docs/`：实验结论、指标解读和规则交接说明。

## 固定业务规则

- 宿舍网络问题属于宿舍设施，部门为慧湖物业中心。
- 校园网络部门为网络管理部。
- 食堂食品安全问题中，只要有人出现身体症状，不论人数多少，紧急度为高。
- 选课系统实际故障因办理时限和后续学习影响，紧急度为高。
- 实验室部分电脑无法启动实验软件，紧急度为中。

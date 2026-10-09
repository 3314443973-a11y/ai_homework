# 校园诉求文本分类（AI 期中作业）

使用中文字符 TF-IDF 与逻辑回归，学习将校园诉求分为宿舍设施、校园网络、食堂餐饮、教学设施、校园安全、其他六类。

当前仓库已有成员 A 的开发版训练脚本和类别预测函数；完整 Web 平台和最终测试成绩仍待团队后续完成。

成员 A 当前模型的交付文件、调用方法和第 5 天测试记录见 [`docs/member_a_day5_handoff.md`](docs/member_a_day5_handoff.md)。

## 目录

```text
ai_homework/
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   ├── __init__.py         # 使 src 可作为 Python 包导入——A
│   ├── train_nb.py         # 使用了朴素贝叶斯（nb），与A的成果对比，选出更好的模型——C
│   ├── train.py            # 开发验证、全量训练与模型保存——A
|   ├── app.py              # Streamlit 页面入口——C
│   └── predict.py          # 给 C 导入的 predict_label(text)——A
│   └── urgency.py          # 紧急度规则，开发使用——B
│   └── department.py       # 部门映射规则，开发使用——B
├── models/
│   ├── README.md           # 模型说明——A
│   ├── label_model_dev.joblib  # 已训练的开发版模型——A
│   └── model_info_dev.json # 数据、参数与版本记录——A
├── model_result/
│   ├── member_a_test_results.csv # 模型用测试集跑出的结果——A
│   └── nb_test_results.csv # C跑出的训练结果——C
├── docs/
│   └── member_a_day5_handoff.md # A 的第 5 天交付与联调说明
└── data/
    ├── README.md
    └── project/
        ├── train_250.csv
        └── test_50.csv
```

## 安装与运行

建议使用 Python 3.11。在本仓库根目录打开终端，执行：

```bash
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux 激活环境：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

仓库已包含开发版模型 `models/label_model_dev.joblib`；安装依赖后即可调用预测函数，正常预测无需训练。若要重新生成当前模型及结果表，在仓库根目录运行 `python -m src.train`。当前脚本用 `data/project/train_250.csv` 的全部 250 条拟合模型，再读取 `data/project/test_50.csv` 的 50 条进行预测，更新模型文件、`model_info_dev.json` 和 `model_result/member_a_test_results.csv`。

训练完成后，在仓库根目录调用预测函数：

```python
from src.predict import predict_label

label = predict_label("宿舍空调不制冷")
print(label)  # 六类中的一个，例如：宿舍设施
```

`predict_label(text)` 接收非空字符串，返回类别字符串；空白输入抛出 `ValueError`，非字符串抛出 `TypeError`。它只负责类别，不返回紧急度、部门或摘要。成员 C 从 GitHub 获取仓库并安装依赖后即可调用，按 `analyze_contract.md` 将类别放入最终结果的 `label` 字段。模型是开发版候选产物，训练脚本和数据一并保留，便于复现；仅加载来源可信的模型文件，并使用 `requirements.txt` 中的依赖版本。

## 后续代码存放与团队整合

**所有成员后续交付的核心功能 Python 代码统一放入 `src/`。** 按功能命名文件，不再套“成员A”“成员B”等个人文件夹。数据继续放在 `data/`。

按现有团队分工，代码交付与整合安排如下；A 的两份脚本已创建，其他文件名仍是建议：

| 成员 | 负责内容 | 建议代码位置 |
| --- | --- | --- |
| A | 主模型训练、保存与加载、类别预测接口 | `src/train.py`、`src/predict.py` |
| B | 数据预处理、紧急度与部门规则 | `src/preprocess.py`、`src/rules.py` |
| C | Naive Bayes 对照、`analyze(text)` 集成与页面 | `src/baseline.py`、`src/analyze.py`、`src/ui.py` |
| D | 模型评估、误判分析与摘要相关功能 | `src/evaluate.py`、`src/summary.py` |

**最后由成员 C（负责集成与页面）在仓库根目录创建并维护 `main.py`，与本 README 同级。** 各成员先提交可调用的函数及输入、输出、依赖和使用示例；C 在模块接口确定后创建统一入口，调用 `src/` 中的功能，其他成员配合联调。无需等所有功能全部完成才开始集成。

`main.py` 负责启动与组织流程，模型训练、预测、规则及摘要等具体实现保留在 `src/` 中。目前 `main.py` 尚未创建，项目还不能通过它启动。最终启动命令由 C 在集成完成后补充到本 README；若使用 Streamlit 页面入口，应写明对应的 Streamlit 启动方式。

各成员在自己的分支提交代码，通过 Pull Request 合并到团队主分支。新增第三方依赖时同步更新 `requirements.txt`，修改公共接口时先与 C 对齐。


## 实验约定

- `train_250.csv` 含 250 条数据，供训练与开发验证使用。
- `test_50.csv` 含 50 条数据，未用于这份模型的 `fit()`；它已用于观察当前模型的测试结果。
- 当前字符 1～3 gram TF-IDF + Logistic Regression 模型用全部 250 条训练数据拟合后，在这 50 条测试数据上答对 **37/50（74%）**，有 13 条误判。结果表见 `model_result/member_a_test_results.csv`。
- 这 50 条的错误已经被查看；若后续据此选特征或参数，同一测试集上的新分数不应称为全新独立测试成绩。旧内部开发验证数字与当前模型测试数字不要混用。

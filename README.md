# 校园诉求智能分流与治理分析平台

AI 期中作业项目：对学生提交的校园诉求进行文本分类，结合紧急度规则和部门映射，提供诉求分流与统计展示。

当前已包含 **六类文本分类模型、紧急度规则、部门映射、统一分析接口和 Streamlit 页面**。页面支持学生提交诉求、查看分析结果，并将记录追加到 CSV；管理员标签页展示总量、高紧急度数量、类别柱状图、紧急度饼图和原始记录。当前是本地演示版本，管理员标签页尚无登录或权限控制。

## 目录结构

```text
ai/
├── README.md
├── requirements.txt                  # 页面运行和模型加载依赖
├── requirements-experiments.txt      # 离线训练与实验依赖
├── .gitignore
├── app.py                            # Streamlit 页面入口 -C
├── src/
│   ├── __init__.py                     #模型初始化
│   ├── analyze.py                    # 组合分类、紧急度和部门结果 -C
│   ├── predict.py                    # predict_label(text) 分类接口 -A
│   ├── predict_copy.py                # 中间测试模型时MLP模型接口 -A
│   ├── train.py                      # TF-IDF + Logistic Regression 训练与评估 -A
│   ├── train_copy.py                      # MLP模型训练训练与评估 -A
│   ├── train_nb.py                   # MultinomialNB 对照实验 -C
│   ├── summary_1.py                  # 初代摘要实现，经测试舍弃 -D
│   ├── summary_2.py                  # 最终摘要实现 -D
│   ├── urgency.py                    # 紧急度规则及解释 -B
│   ├── department.py                 # 类别到部门的固定映射 -B
│   ├── evaluate_urgency.py            # 紧急度规则评估 -B
│   ├── tfidf_experiment.py            # 分类特征对照实验 -B
│   ├── urgency_tfidf_experiment.py    # 紧急度 TF-IDF 对照实验 -B
│   └── verify_environment.py          # 旧环境自检
│   ├── test_rules.py                 # 规则自检程序，不参与训练 -B 
├── data/
│   ├── README.md
│   ├── history.csv                   # 页面历史记录 -C
│   └── project/
│       ├── README.md                 # 历史交接说明
│       ├── train_250.csv             # 250 条训练数据
│       └── test_50.csv               # 50 条交叉验证集数据
│       └── test_50_ultimate.csv      # 50 条测试数据
│       └── long_text_51.csv          # 51 条长文本数据 -B
├── models/   #全部来自A
│   ├── README.md
│   ├── label_model_dev.joblib        # 当前 predict.py 实际加载的模型
│   ├── model_info_dev.json           # 当前模型参数、版本、数据哈希与成绩
│   ├── model_info_dev22.json         # MLP模型参数、版本、数据哈希与成绩
│   ├── label_model_dev22.joblib      # MLP模型predict_copy.py 实际加载的模型
├── model_result/
│   ├── member_a_test_results.csv     # 逻辑回归逐条预测结果 -A
│   ├── nb_test_results.csv           # 朴素贝叶斯逐条预测结果
│   ├── predictions.csv
│   ├── 分词对紧急度影响测试结果说明.md
│   └── urgency_results/              # 已保存的规则评估报告与逐条结果
├── summary/
│   ├── README.md                     # 摘要模块文件说明 -D
│   ├── analyze_summary_patterns.py   # 分析词组和字频，生成频率表，辅助摘要规则编写 -D
│   ├── evaluate_summary.py           # 初代摘要评估脚本，保存评估结果 -D
│   ├── summary_evaluation_results.csv # 初代摘要评估结果 -D
│   ├── summary_phrase_frequency.csv  # 词组频率统计 -D
│   ├── summary_word_frequency.csv    # 字频统计 -D
│   └── test_summary.py               # 初代摘要小测试集 -D
├── rules/
│   ├── analyze_contract.md           # 目标接口约定
│   ├── label_guide.md                # 标注规范
├── verify_environment.py             # 根目录保留的旧环境自检脚本
├── summary.json                      # 根目录保留的紧急度评估汇总
└── confusion_matrix.csv              # 根目录保留的混淆矩阵
```

## 安装与启动

以下命令均在 **ai 仓库根目录** 执行。建议使用 Python **3.13** 创建虚拟环境，Windows x64 下也已验证 Python **3.11** 可解析整份依赖清单。

### 1. 创建并激活虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux 激活命令：

```bash
source .venv/bin/activate
```

### 2. 安装依赖

```bash
python -m pip install -r requirements.txt
```

此命令安装分类模型和 Streamlit 页面所需的依赖。Streamlit Community Cloud 读取此清单，模型依赖版本与 `models/model_info_dev.json` 中的记录一致。

运行离线实验、评估图表、中文分词或 TensorFlow 训练时，可用 Python 3.13 安装完整依赖：

```bash
python -m pip install -r requirements-experiments.txt
```

### 3. 启动页面

```bash
python -m streamlit run app.py
```

打开终端显示的本地地址即可使用。

- **学生端**：输入不超过 200 字的诉求，点击“提交并分析”，查看类别、紧急度和负责部门。
- **管理员端**：读取 `data/history.csv` 并展示统计图表与记录。

页面按相对路径读写历史记录，因此必须从仓库根目录启动。首次使用时确保 `data/history.csv` 存在，且第一行为：

```csv
content,label,urgency,department
```

当前代码仅处理空 CSV，未处理文件不存在的情况；追加记录时也不会自动写入表头。管理员统计在页面执行时读取，新增记录后如未更新，可重新运行或刷新页面。

## 核心接口

### 类别预测

在仓库根目录的 Python 中调用：

```python
from src.predict import predict_label

label = predict_label("宿舍空调不制冷")
print(label)
```

`predict_label(text)` 接收非空字符串，返回六类之一。非字符串抛出 `TypeError`，空白字符串抛出 `ValueError`；缺少模型时抛出 `FileNotFoundError`。只需要分类时，无需安装 Streamlit 和 Plotly。

### 统一分析

从仓库根目录验证统一分析接口：

```bash
python -c "from src.analyze import analyze; print(analyze('宿舍空调不制冷'))"
```

成功时返回以下结构（类别以实际模型预测为准）：

```python
{
    "summary": "宿舍,空调故障",
    "label": "宿舍设施",
    "urgency": "中",
    "department": "慧湖物业中心",
    "error": None,
}
```

`analyze(text)` 成功时返回 `error: None`，失败时抛出异常，由调用方处理。

### 紧急度与部门

```python
from src.urgency import get_urgency, analyze_urgency
from src.department import get_department

print(get_urgency("教室插座正在冒烟"))  # 高
print(analyze_urgency("教室插座正在冒烟"))  # 包含原因、命中规则及复核标记
print(get_department("校园安全"))  # 保卫处
```

紧急度分为 **高、中、低**，通过规则判断；未命中明确高低规则的文本默认返回“中”，并设置 `needs_review=True`。当前页面仅显示紧急度，未展示规则解释和复核标记。

| 类别 | 负责部门 |
| --- | --- |
| 宿舍设施 | 慧湖物业中心 |
| 校园网络 | 网络管理部 |
| 食堂餐饮 | 餐饮管理 |
| 教学设施 | 教学保障 |
| 校园安全 | 保卫处 |
| 其他 | 人工处理 |

部门由最终预测类别映射得到；未知类别会抛出 `ValueError`。宿舍网络归属等标注边界见 [rules/label_guide.md](rules/label_guide.md)。

## 模型训练与结果

### 逻辑回归主模型

```bash
python -m src.train
```

使用字符级 TF-IDF（`analyzer="char"`、`ngram_range=(1, 3)`）与 `LogisticRegression(max_iter=1000)`，用全部 250 条训练数据拟合，再对 50 条测试数据预测。训练会覆盖：

- `models/label_model_dev.joblib`
- `models/model_info_dev.json`
- `model_result/member_a_test_results.csv`

正常页面预测直接加载已有模型，无需每次训练。

### 朴素贝叶斯对照

```bash
python -m src.train_nb
```

使用字符级 TF-IDF 2～3 gram 与 `MultinomialNB`，打印分类报告、误判样本和混淆矩阵，覆盖 `model_result/nb_test_results.csv`。**运行前须将脚本中的两个 `D:\kuyly\code\self_program\ai\...` 数据绝对路径改为本机路径**；当前脚本未保存可供页面切换的 NB 模型。

### 当前保存的结果

下表按现有逐条结果 CSV 与 JSON 汇总核对，未重新训练：

| 方案 | 数据规模 | 准确率 | 说明 |
| --- | --- | --- | --- |
| TF-IDF 1～3 gram + 逻辑回归 | 50 条 | 94%（47/50） | 当前页面分类模型对应记录，13 条误判 |
| TF-IDF 2～3 gram + 朴素贝叶斯 | 50 条 | 68%（34/50） | 当前 NB 结果表，16 条误判 |
| 紧急度规则 | 50 条 | 100%（50/50） | 已保存报告 Macro F1 为 1.0，16 条标记需复核 |

分类准确率与紧急度准确率衡量不同任务，不能相互替代。测试数据已用于观察误判和规则调整；继续据此改进后的成绩不代表全新独立测试表现，也不能据此推断真实场景达到相同准确率。旧交接材料中的 78% 等数字属于其他实验记录，不作为当前加载模型的成绩。

## 规则测试与实验脚本

运行已有回归测试：

```bash
python -m unittest discover -s rules -p "test_rules.py"
```

部分保留脚本尚未完成目录迁移，不能直接视为当前项目的可用入口：

| 文件 | 当前需要对齐的内容 |
| --- | --- |
| `src/evaluate_urgency.py` | 输入仍为 `data/test_50.csv`，应对齐 `data/project/test_50.csv`；输出仍指向 `results/urgency_rules/` |
| `src/tfidf_experiment.py` | 输入仍在 `data/`，输出指向 `results/tfidf_jieba/` |
| `src/urgency_tfidf_experiment.py` | 输入仍在 `data/`，输出指向 `results/urgency_tfidf/`；仅为离线实验，不替换页面紧急度规则 |
| 根目录及 `src/` 下的 `verify_environment.py` | 固定检查旧 Python / 依赖版本，并引用不存在的 `model/complaint_label_classifier_char_1_3.joblib`；根目录副本的根路径计算也不适用于当前位置 |

已有紧急度评估文件位于 `model_result/urgency_results/`；这些保存位置与评估脚本当前输出配置并不一致。历史交接文档中提到的 `docs/`、`src/build_final_model.py` 等文件当前不存在，请以本目录实际代码和本 README 为准。

## 协作与后续完善

核心 Python 功能统一放入 `src/`，数据放入 `data/`，模型放入 `models/`，评估产物放入 `model_result/`。现有职责对应关系：

| 成员 | 当前对应内容 |
| --- | --- |
| A | `train.py`、`predict.py`、主模型及测试结果 |
| B | `urgency.py`、`department.py`、规则测试与特征实验 |
| C | `train_nb.py`、`analyze.py`、`app.py` 与页面集成 |
| D | 按团队分工继续补充评估、误判分析与摘要功能；当前未见独立摘要实现 |

后续重点：统一实验路径与模型记录，完善历史文件初始化、异常处理与页面统计刷新，实现摘要并对齐接口约定，将需人工复核的结果展示给管理员。

各成员通过分支和 Pull Request 合并代码。新增依赖同步维护 `requirements.txt`，公共字段统一使用 `label`。`.gitignore` 当前忽略 `*.joblib`、`*.docx` 等文件，首次克隆时应确认模型是否随仓库提供；缺少当前模型时运行 `python -m src.train` 生成。

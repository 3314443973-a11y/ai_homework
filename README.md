# 校园诉求智能分流与治理分析平台

AI 期中作业项目：对学生提交的校园诉求进行文本分类，结合紧急度规则和部门映射，提供诉求分流与统计展示。

当前已包含 **六类文本分类模型、紧急度规则、部门映射、统一分析接口和 Streamlit 页面**。页面支持学生提交诉求、查看分析结果，并将记录追加到 CSV；管理员标签页展示总量、高紧急度数量、类别柱状图、紧急度饼图和原始记录。当前是本地演示版本，管理员标签页尚无登录或权限控制。

## 目录结构

```text
ai/
├── README.md
├── requirements.txt                  # 当前核心模型依赖（页面与实验依赖需另装）
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── app.py                        # Streamlit 页面入口 -C
│   ├── analyze.py                    # 组合分类、紧急度和部门结果 -C
│   ├── predict.py                    # predict_label(text) 分类接口 -C
│   ├── train.py                      # TF-IDF + Logistic Regression 训练与评估 -A
│   ├── train_nb.py                   # MultinomialNB 对照实验 -C
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
│       └── test_50.csv               # 50 条测试数据
├── models/
│   ├── README.md
│   ├── label_model_dev.joblib        # 当前 predict.py 实际加载的模型
│   ├── model_info_dev.json           # 当前模型参数、版本、数据哈希与成绩
│   └── category_model_dev.joblib     # 保留文件，当前预测接口未使用
├── model_result/
│   ├── member_a_test_results.csv     # 逻辑回归逐条预测结果
│   ├── nb_test_results.csv           # 朴素贝叶斯逐条预测结果
│   ├── predictions.csv
│   ├── 分词对紧急度影响测试结果说明.md
│   └── urgency_results/              # 已保存的规则评估报告与逐条结果
├── rules/
│   ├── analyze_contract.md           # 目标接口约定
│   ├── label_guide.md                # 标注规范
│   └── test_rules.py                 # 紧急度与部门映射回归测试
├── verify_environment.py             # 根目录保留的旧环境自检脚本
├── summary.json                      # 根目录保留的紧急度评估汇总
└── confusion_matrix.csv              # 根目录保留的混淆矩阵
```

## 安装与启动

以下命令均在 **ai 仓库根目录** 执行。当前模型元数据记录的训练环境为 Python **3.13.13**；旧交接说明中的 Python 3.11.16 与当前模型记录不同。

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
python -m pip install streamlit plotly
```

`requirements.txt` 当前仅包含 pandas、scikit-learn、joblib、numpy 和 scipy；第二条命令补充页面依赖。运行朴素贝叶斯或特征对照实验时，还需安装：

```bash
python -m pip install matplotlib jieba
```

模型元数据中的 numpy / scipy 版本与当前依赖清单不同。若加载模型出现兼容性问题，可在目标环境执行下文训练命令，重新生成模型。已有 `.venv` 若失效，应使用本机可用的 Python 重新创建环境。

### 3. 启动页面

```bash
python -m streamlit run src/app.py
```

打开终端显示的本地地址即可使用。当前入口为 `src/app.py`，无需创建 `main.py`。

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

`src/analyze.py` 当前使用同目录绝对导入。Streamlit 入口可直接调用；从仓库根目录进行命令行联调时可使用：

```bash
python -c "import sys; sys.path.insert(0, 'src'); from analyze import analyze; print(analyze('宿舍空调不制冷'))"
```

成功时返回以下结构（类别以实际模型预测为准）：

```python
{
    "label": "宿舍设施",
    "urgency": "中",
    "department": "慧湖物业中心",
    "error": None,
}
```

当前 `analyze(text)` 不捕获异常，失败由调用方处理；`error` 字段仅在成功时返回 `None`。目标约定 [rules/analyze_contract.md](rules/analyze_contract.md) 中的 `summary` 尚未实现，当前接口也未返回置信度。

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
| TF-IDF 1～3 gram + 逻辑回归 | 50 条 | 74%（37/50） | 当前页面分类模型对应记录，13 条误判 |
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

后续重点：补齐并统一依赖版本；统一实验路径与模型记录；完善历史文件初始化、异常处理与页面统计刷新；实现摘要并对齐接口约定；将需人工复核的结果展示给管理员。

各成员通过分支和 Pull Request 合并代码。新增依赖同步维护 `requirements.txt`，公共字段统一使用 `label`。`.gitignore` 当前忽略 `*.joblib`、`*.docx` 等文件，首次克隆时应确认模型是否随仓库提供；缺少当前模型时运行 `python -m src.train` 生成。

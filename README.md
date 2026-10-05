# 校园诉求文本分类（AI 期中作业）

使用中文字符 TF-IDF 与逻辑回归，学习将校园诉求分为宿舍设施、校园网络、食堂餐饮、教学设施、校园安全、其他六类。

当前仓库包含项目数据和基础目录，src 目录预留用于后续核心功能代码；尚未提供完整 Web 平台、训练或预测实现及最终测试成绩。

## 目录

```text
campus-request-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── src/                   # 后续核心功能代码
│   └── .gitkeep            # 保留空目录，加入代码后可删除
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

目前尚无可运行的功能代码，依赖清单暂留空。后续成员添加实际依赖后，可执行：

```bash
python -m pip install -r requirements.txt
```

## 后续代码存放与团队整合

**所有成员后续交付的核心功能 Python 代码统一放入 `src/`。** 按功能命名文件，不再套“成员A”“成员B”等个人文件夹。数据继续放在 `data/`。`src/.gitkeep` 用于保留空目录，加入代码后可删除。

按现有团队分工，代码交付与整合安排如下；下列文件名是建议，尚未创建：

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
- `test_50.csv` 含 50 条数据，保留用于最终测试，不用于学习词表或选择参数。
- 开发验证准确率不代表最终测试成绩。当前未发布最终测试指标。
- CSV 的 `urgency` 与 `department` 是附加标注字段，当前分类练习的目标是 `label`。

## 本次整理范围

保留现有项目数据及基础配置，预留 src 代码目录；目录统一使用英文名称。原工作目录中的学习资料、个人学习计划、Word 方案、PDF 清单、讲义生成工具、另一标注版本的练习数据、数据C、缓存与 Git 历史未打包。

数据来源与授权信息尚未在现有文件中明确，详见 `data/README.md`。本整理版未指定开源许可证。

## 上传方式

将本目录作为仓库根目录，使 GitHub 首页直接显示本 README、依赖清单和代码目录。不要将外层 `github-upload` 一起作为项目目录上传。本次整理仅创建本地文件夹，没有推送到远程仓库。

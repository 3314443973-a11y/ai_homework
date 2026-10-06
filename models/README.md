# 模型文件

仓库包含可直接用于联调的 `category_model_dev.joblib` 和 `model_info_dev.json`。安装根目录 `requirements.txt` 中的依赖后，即可通过 `from src.predict import predict_category` 调用。需要重新生成时，在仓库根目录运行 `python -m src.train`。

`category_model_dev.joblib` 是使用训练文件全部 250 条数据生成的**开发版**模型，供成员 C 联调。`model_info_dev.json` 记录训练数据指纹、随机种子、软件版本及内部开发验证结果。两者均纳入这次交付，使队友下载代码后无需先训练。模型仍可从仓库内的训练脚本和数据重新生成。

`joblib` 加载时会反序列化 Python 对象，只加载来源可信的文件；使用与 `requirements.txt` 一致的依赖版本。若之后改变训练代码、训练数据或依赖版本，应重新生成并同步模型与版本记录。

最终模型与测试结果确定后，再决定是否把最终模型文件纳入仓库或单独交付。

"""Label prediction entrypoint for member C's integration code."""

from __future__ import annotations

from pathlib import Path

from joblib import load

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "label_model_dev22.joblib"


def _load_bundle():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到开发版模型：{MODEL_PATH}。请先在仓库根目录运行 python 'src/train_copy.py'。"
        )
    return load(MODEL_PATH)


def predict_label(text: str) -> str:
    """Return one of the six labels for a non-empty complaint."""
    if not isinstance(text, str):
        raise TypeError("诉求必须是字符串。")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("诉求内容不能为空。")
    bundle=_load_bundle()
    features=bundle["vectorizer"].transformer([cleaned]).toarray()  #先用向量器处理文本得到我们需要的格式
    probabilities = bundle["model"].predict(features, verbose=0)    #猜测不同类别的概率
    label_id = int(probabilities[0].argmax())
    return str(bundle["labels"][label_id])
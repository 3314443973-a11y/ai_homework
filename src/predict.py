"""Category prediction entrypoint for member C's integration code."""

from __future__ import annotations

from pathlib import Path

from joblib import load

#确认存储路径
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "category_model_dev.joblib"


#调用joblib中存的模型
def _load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"找不到开发版模型：{MODEL_PATH}。请先在仓库根目录运行 python -m src.train。"
        )
    return load(MODEL_PATH)


#调用模型本体
def predict_category(text: str) -> str:
    """Return one of the six category names for a non-empty complaint."""
    if not isinstance(text, str):
        raise TypeError("诉求必须是字符串。")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("诉求内容不能为空。")
    return str(_load_model().predict([cleaned])[0])

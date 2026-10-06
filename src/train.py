"""Train the development category model from data/project/train_250.csv.

Run from the repository root with: python -m src.train
The reserved test_50.csv is not read here.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "project" / "train_250.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "category_model_dev.joblib"
INFO_PATH = MODEL_DIR / "model_info_dev.json"
LABELS = ("宿舍设施", "校园网络", "食堂餐饮", "教学设施", "校园安全", "其他")
REQUIRED_COLUMNS = ("text", "label", "urgency", "department")
DEV_RANDOM_STATE = 6


def make_model() -> Pipeline:
    """Keep text vectorization and classification together."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 3))),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


def load_training_data() -> tuple[pd.Series, pd.Series, str]:
    if not TRAIN_CSV.is_file():
        raise FileNotFoundError(f"找不到项目训练数据：{TRAIN_CSV}")

    data_hash = hashlib.sha256(TRAIN_CSV.read_bytes()).hexdigest()
    frame = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"训练文件缺少字段：{missing}")

    texts = frame["text"].str.strip()
    labels = frame["label"].str.strip()
    empty_rows = frame.index[texts.eq("")].tolist()
    invalid_rows = frame.index[~labels.isin(LABELS)].tolist()
    duplicate_rows = frame.index[texts.ne("") & texts.duplicated(keep=False)].tolist()
    if empty_rows or invalid_rows or duplicate_rows:
        raise ValueError(
            "训练数据需要先复核："
            f"空文本索引={empty_rows}，无效类别索引={invalid_rows}，"
            f"完全重复文本索引={duplicate_rows}"
        )
    return texts, labels, data_hash


def main() -> None:
    texts, labels, data_hash = load_training_data()
    X_dev_train, X_dev_valid, y_dev_train, y_dev_valid = train_test_split(
        texts, labels, test_size=0.2, random_state=DEV_RANDOM_STATE, stratify=labels
    )

    validation_model = make_model()
    validation_model.fit(X_dev_train, y_dev_train)
    predictions = validation_model.predict(X_dev_valid)
    correct = int((predictions == y_dev_valid.to_numpy()).sum())
    dev_accuracy = float(accuracy_score(y_dev_valid, predictions))

    # The saved candidate learns from all 250 rows in the training file.
    # Its independent final-test accuracy has not been measured here.
    candidate_model = make_model()
    candidate_model.fit(texts, labels)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate_model, MODEL_PATH)

    info = {
        "status": "开发版候选模型；尚未在保留测试集上评估",
        "training_source": "data/project/train_250.csv",
        "training_sha256": data_hash,
        "candidate_training_rows": len(texts),
        "development_split": {
            "train": len(X_dev_train),
            "validation": len(X_dev_valid),
            "random_state": DEV_RANDOM_STATE,
        },
        "development_validation": {
            "correct": correct,
            "total": len(X_dev_valid),
            "accuracy": dev_accuracy,
        },
        "score_note": "开发准确率属于内部划分模型，不能作为全量重训候选模型的独立测试成绩。",
        "features": {"analyzer": "char", "ngram_range": [2, 3]},
        "classifier": "LogisticRegression(max_iter=1000)",
        "python": platform.python_version(),
        "packages": {
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
    }
    INFO_PATH.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"训练数据：{TRAIN_CSV}（{len(texts)} 条）")
    print(f"开发划分：{len(X_dev_train)} 条训练 / {len(X_dev_valid)} 条验证；随机种子 {DEV_RANDOM_STATE}")
    print(f"开发验证：{correct}/{len(X_dev_valid)}，准确率 {dev_accuracy:.1%}")
    print(f"已保存 250 条训练数据重训的开发版模型：{MODEL_PATH}")
    print(f"版本记录：{INFO_PATH}")


if __name__ == "__main__":
    main()

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

#定位文件地址，确定标签和随即生成数
ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "project" / "train_250.csv"
TEST_CSV = ROOT / "data" / "project" / "test_50.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "category_model_dev.joblib"
INFO_PATH = MODEL_DIR / "model_info_dev.json"
LABELS = ("宿舍设施", "校园网络", "食堂餐饮", "教学设施", "校园安全", "其他")
REQUIRED_COLUMNS = ("text", "label", "urgency", "department")
DEV_RANDOM_STATE = 6


#制作模型的组装函数
def make_model() -> Pipeline:
    """Keep text vectorization and classification together."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 3))),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])


#获取训练数据
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

#获取测试数据
def load_test_data() -> tuple[pd.Series, pd.Series, str]:
    if not TEST_CSV.is_file():
        raise FileNotFoundError(f"找不到项目训练数据：{TEST_CSV}")

    data_hash_test = hashlib.sha256(TEST_CSV.read_bytes()).hexdigest()
    frame_test = pd.read_csv(TEST_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing_test = [column for column in REQUIRED_COLUMNS if column not in frame_test.columns]
    if missing_test:
        raise ValueError(f"训练文件缺少字段：{missing_test}")

    texts_test = frame_test["text"].str.strip()
    labels_test = frame_test["label"].str.strip()
    empty_rows_test = frame_test.index[texts_test.eq("")].tolist()
    invalid_rows_test = frame_test.index[~labels_test.isin(LABELS)].tolist()
    duplicate_rows_test = frame_test.index[texts_test.ne("") & texts_test.duplicated(keep=False)].tolist()
    if empty_rows_test or invalid_rows_test or duplicate_rows_test:
        raise ValueError(
            "训练数据需要先复核："
            f"空文本索引={empty_rows_test}，无效类别索引={invalid_rows_test}，"
            f"完全重复文本索引={duplicate_rows_test}"
        )
    return texts_test, labels_test, data_hash_test

#生成模型joblib和json以及跑测试数据（测试数据是基于250条出的正确率，没用最后的50条
def main() -> None:
    texts, labels, data_hash = load_training_data()
    texts_test, labels_test, data_hash_test = load_test_data()
    X_dev_train, X_dev_valid, y_dev_train, y_dev_valid = train_test_split(
        texts, labels, test_size=0.2, random_state=DEV_RANDOM_STATE, stratify=labels
    )

    candidate_model = make_model()
    candidate_model.fit(texts, labels)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate_model, MODEL_PATH)
    predictions = candidate_model.predict(texts_test)
    correct = int((predictions == labels_test.to_numpy()).sum())
    dev_accuracy = float(accuracy_score(labels_test, predictions))
    preview = pd.DataFrame({
    "原训练表索引": texts_test.index,
    "诉求": texts_test.to_numpy(),
    "真实类别": labels_test.to_numpy(),
    "预测类别": predictions,
    })
    preview.to_csv(ROOT / "model_result" / "member_a_test_results.csv", index=False, encoding="utf-8-sig")

#生成json
    info = {
        "status": "开发版候选模型；尚未在保留测试集上评估",
        "training_source": "data/project/train_250.csv",
        "training_sha256": data_hash,
        "candidate_training_rows": len(texts),
      
        "test_result": {
            "correct": correct,
            "total": len(texts_test),
            "accuracy": dev_accuracy,
        },
        "score_note": "开发准确率属于内部划分模型，不能作为全量重训候选模型的独立测试成绩。",
        "features": {"analyzer": "char", "ngram_range": [1, 3]},
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
    print(f"开发验证：{correct}/{len(texts_test)}，准确率 {dev_accuracy:.1%}")
    print(f"已保存 250 条训练数据训的开发版模型：{MODEL_PATH}")
    print(f"版本记录：{INFO_PATH}")


#确保只有在终端输入python -m src.train才会重新训练模型
if __name__ == "__main__":
    main()

import csv
import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train_250.csv"
TEST_PATH = ROOT / "data" / "test_50.csv"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "complaint_label_classifier_char_1_3.joblib"
SUMMARY_PATH = MODEL_DIR / "model_summary.json"
RANDOM_STATE = 20261002


def read_dataset(path):
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            rows = list(csv.DictReader(raw.decode(encoding).splitlines()))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeError(f"无法识别 {path.name} 的编码")

    expected = ["text", "label", "urgency", "department"]
    if not rows or list(rows[0]) != expected:
        raise ValueError(f"{path.name} 字段必须为 {expected}")
    return rows


def main():
    train_rows = read_dataset(TRAIN_PATH)
    test_rows = read_dataset(TEST_PATH)
    if len(train_rows) != 250 or len(test_rows) != 50:
        raise ValueError("训练集必须为250条，测试集必须为50条")

    train_texts = [row["text"] for row in train_rows]
    train_labels = [row["label"] for row in train_rows]
    test_texts = [row["text"] for row in test_rows]
    test_labels = [row["label"] for row in test_rows]
    if set(train_texts) & set(test_texts):
        raise ValueError("训练集和测试集存在文本交集")

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(1, 3),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight=None,
                ),
            ),
        ]
    )
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts)

    summary = {
        "model": "Complaint label classifier: TF-IDF char 1-3 gram + Logistic Regression",
        "python_version": sys.version.split()[0],
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "feature_count": len(pipeline.named_steps["tfidf"].get_feature_names_out()),
        "accuracy": accuracy_score(test_labels, predictions),
        "macro_f1": f1_score(test_labels, predictions, average="macro", zero_division=0),
        "mistake_count": sum(
            true_label != predicted_label
            for true_label, predicted_label in zip(test_labels, predictions)
        ),
        "random_state": RANDOM_STATE,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()

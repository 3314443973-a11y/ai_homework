"""Compare character TF-IDF ranges for urgency classification.

This is an offline experiment only. The production urgency implementation in
urgency.py remains rule based and is not replaced by these models.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "urgency_tfidf"
TRAIN_PATH = DATA_DIR / "train_250.csv"
TEST_PATH = DATA_DIR / "test_50.csv"
URGENCY_ORDER = ["高", "中", "低"]
RANDOM_STATE = 20261002

EXPERIMENTS = {
    "char_1_2": (1, 2),
    "char_1_3": (1, 3),
    "char_2_3": (2, 3),
    "char_2_4": (2, 4),
}


def read_dataset(path: Path) -> list[dict[str, str]]:
    raw = path.read_bytes()
    decoded = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise UnicodeError(f"无法识别 {path.name} 的编码")

    rows = list(csv.DictReader(decoded.splitlines()))
    expected = ["text", "label", "urgency", "department"]
    if not rows or list(rows[0]) != expected:
        raise ValueError(f"{path.name} 字段必须为 {expected}")
    invalid = sorted({row["urgency"] for row in rows} - set(URGENCY_ORDER))
    if invalid:
        raise ValueError(f"{path.name} 含非法紧急度：{invalid}")
    return rows


def save_confusion_matrix(matrix: np.ndarray, experiment_id: str) -> None:
    csv_path = RESULTS_DIR / f"confusion_matrix_{experiment_id}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["真实\\预测", *URGENCY_ORDER])
        for urgency, values in zip(URGENCY_ORDER, matrix.tolist()):
            writer.writerow([urgency, *values])

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(URGENCY_ORDER)),
        yticks=np.arange(len(URGENCY_ORDER)),
        xticklabels=URGENCY_ORDER,
        yticklabels=URGENCY_ORDER,
        xlabel="预测紧急度",
        ylabel="真实紧急度",
        title=f"紧急度混淆矩阵 {experiment_id}",
    )
    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(csv_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    train_rows = read_dataset(TRAIN_PATH)
    test_rows = read_dataset(TEST_PATH)
    if len(train_rows) != 250 or len(test_rows) != 50:
        raise ValueError("实验固定使用250条训练集和50条旧开发测试集")

    train_texts = [row["text"] for row in train_rows]
    train_targets = [row["urgency"] for row in train_rows]
    test_texts = [row["text"] for row in test_rows]
    test_targets = [row["urgency"] for row in test_rows]
    if set(train_texts) & set(test_texts):
        raise ValueError("训练集和旧开发测试集存在文本交集")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    details = {}

    for experiment_id, ngram_range in EXPERIMENTS.items():
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=ngram_range,
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

        start = time.perf_counter()
        pipeline.fit(train_texts, train_targets)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predictions = pipeline.predict(test_texts)
        predict_seconds = time.perf_counter() - start

        vectorizer = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["classifier"]
        feature_names = vectorizer.get_feature_names_out()
        top_features = {}
        for class_index, urgency in enumerate(classifier.classes_):
            indices = np.argsort(classifier.coef_[class_index])[-15:][::-1]
            top_features[urgency] = [
                {
                    "feature": feature_names[index],
                    "weight": float(classifier.coef_[class_index, index]),
                }
                for index in indices
            ]

        matrix = confusion_matrix(test_targets, predictions, labels=URGENCY_ORDER)
        mistakes = [
            {
                "text": text,
                "label": row["label"],
                "true_urgency": true,
                "predicted_urgency": predicted,
            }
            for text, row, true, predicted in zip(
                test_texts, test_rows, test_targets, predictions
            )
            if true != predicted
        ]
        result = {
            "experiment_id": experiment_id,
            "ngram_range": list(ngram_range),
            "feature_count": len(feature_names),
            "accuracy": accuracy_score(test_targets, predictions),
            "macro_f1": f1_score(
                test_targets, predictions, average="macro", zero_division=0
            ),
            "mistake_count": len(mistakes),
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
            "classification_report": classification_report(
                test_targets,
                predictions,
                labels=URGENCY_ORDER,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": matrix.tolist(),
            "top_features": top_features,
            "mistakes": mistakes,
        }
        details[experiment_id] = result
        summary.append(
            {
                "experiment_id": experiment_id,
                "ngram_range": f"{ngram_range[0]}-{ngram_range[1]}",
                "feature_count": result["feature_count"],
                "accuracy": round(result["accuracy"], 6),
                "macro_f1": round(result["macro_f1"], 6),
                "mistake_count": result["mistake_count"],
                "train_seconds": round(train_seconds, 6),
                "predict_seconds": round(predict_seconds, 6),
            }
        )
        save_confusion_matrix(matrix, experiment_id)

    summary.sort(
        key=lambda item: (-item["macro_f1"], -item["accuracy"], item["feature_count"])
    )
    with (RESULTS_DIR / "experiment_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    (RESULTS_DIR / "experiment_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metadata = {
        "python_version": sys.version.split()[0],
        "train_path": "data/train_250.csv",
        "test_path": "data/test_50.csv",
        "final_test_used": False,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_urgency_distribution": dict(Counter(train_targets)),
        "test_urgency_distribution": dict(Counter(test_targets)),
        "classifier": "LogisticRegression(max_iter=2000, class_weight=None)",
        "fixed_vectorizer_settings": {"min_df": 2, "sublinear_tf": True},
        "note": "本实验只比较字级范围，不替换生产环境中的规则式 urgency.py。",
    }
    (RESULTS_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"ranking": summary, **metadata}, ensure_ascii=True))


if __name__ == "__main__":
    main()

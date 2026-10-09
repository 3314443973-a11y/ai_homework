import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "test_50.csv"
RESULTS_DIR = ROOT / "results" / "urgency_rules"
LABELS = ["高", "中", "低"]
sys.path.insert(0, str(ROOT / "src"))

from urgency import analyze_urgency  # noqa: E402


def read_rows(path):
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return list(csv.DictReader(raw.decode(encoding).splitlines()))
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别 {path.name} 的编码")


def main():
    rows = read_rows(DATA_PATH)
    actual = []
    predicted = []
    details = []
    for row in rows:
        result = analyze_urgency(row["text"])
        actual.append(row["urgency"])
        predicted.append(result.urgency)
        details.append(
            {
                "text": row["text"],
                "label": row["label"],
                "actual_urgency": row["urgency"],
                "predicted_urgency": result.urgency,
                "matched_rule": result.matched_rule,
                "reason": result.reason,
                "needs_review": result.needs_review,
                "is_correct": row["urgency"] == result.urgency,
            }
        )

    matrix = [[0 for _ in LABELS] for _ in LABELS]
    label_index = {label: index for index, label in enumerate(LABELS)}
    for actual_value, predicted_value in zip(actual, predicted):
        matrix[label_index[actual_value]][label_index[predicted_value]] += 1

    per_label = {}
    f1_values = []
    for index, label in enumerate(LABELS):
        true_positive = matrix[index][index]
        false_positive = sum(matrix[row][index] for row in range(len(LABELS)) if row != index)
        false_negative = sum(matrix[index][column] for column in range(len(LABELS)) if column != index)
        support = sum(matrix[index])
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }
    correct_count = sum(matrix[index][index] for index in range(len(LABELS)))
    report = {
        "test_rows": len(rows),
        "accuracy": correct_count / len(rows),
        "macro_f1": sum(f1_values) / len(f1_values),
        "actual_distribution": dict(Counter(actual)),
        "predicted_distribution": dict(Counter(predicted)),
        "classification_report": per_label,
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": matrix,
        "mismatch_count": sum(not item["is_correct"] for item in details),
        "review_count": sum(item["needs_review"] for item in details),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (RESULTS_DIR / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    with (RESULTS_DIR / "mismatches.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(item for item in details if not item["is_correct"])
    with (RESULTS_DIR / "confusion_matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["真实\\预测", *LABELS])
        for label, values in zip(LABELS, matrix):
            writer.writerow([label, *values])
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()

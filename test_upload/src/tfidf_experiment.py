import csv
import json
import sys
import time
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

try:
    import jieba
except ImportError:
    jieba = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "tfidf_jieba"
TRAIN_PATH = DATA_DIR / "train_250.csv"
TEST_PATH = DATA_DIR / "test_50.csv"
LABEL_ORDER = ["宿舍设施", "校园网络", "食堂餐饮", "教学设施", "校园安全", "其他"]
RANDOM_STATE = 20261002

DOMAIN_WORDS = {
    "宿舍", "寝室", "宽带", "断网", "网络", "空调", "洗衣机", "水龙头", "热水",
    "卫生间", "淋浴间", "物业中心", "慧湖物业中心", "校园网", "无线网络", "认证",
    "教学楼", "图书馆", "实验楼", "体育馆", "教务系统", "食堂", "窗口", "饭菜",
    "食品", "卫生", "餐具", "排队", "扣费", "教室", "投影仪", "投影", "课桌",
    "座椅", "电子白板", "实验室", "自习室", "插座", "照明", "楼梯", "扶手",
    "路灯", "电线", "漏电", "冒烟", "火花", "消防通道", "安全出口", "保卫处",
    "失物招领", "快递站", "奖学金", "心理咨询", "社团活动", "考试周", "无法",
    "不能", "没有", "一直", "经常", "建议", "希望", "危险", "维修", "故障",
}
MAX_DOMAIN_WORD_LENGTH = max(map(len, DOMAIN_WORDS))


def dictionary_tokenize(text):
    """Offline longest-match tokenizer using a fixed campus-domain lexicon."""
    tokens = []
    index = 0
    while index < len(text):
        character = text[index]
        if character.isspace() or character in "，。！？、,.!?;；:：()（）\"'":
            index += 1
            continue
        if character.isascii() and character.isalnum():
            end = index + 1
            while end < len(text) and text[end].isascii() and text[end].isalnum():
                end += 1
            tokens.append(text[index:end].lower())
            index = end
            continue
        matched = None
        upper = min(len(text), index + MAX_DOMAIN_WORD_LENGTH)
        for end in range(upper, index, -1):
            candidate = text[index:end]
            if candidate in DOMAIN_WORDS:
                matched = candidate
                break
        if matched:
            tokens.append(matched)
            index += len(matched)
        else:
            tokens.append(character)
            index += 1
    return tokens


WORD_TOKENIZER = jieba.lcut if jieba is not None else dictionary_tokenize
WORD_TOKENIZER_NAME = "jieba.lcut" if jieba is not None else "offline_domain_dictionary_longest_match"

EXPERIMENTS = {
    "word_1_2": {
        "description": "词级 TF-IDF，1-2 gram",
        "vectorizer": {
            "analyzer": "word",
            "tokenizer": WORD_TOKENIZER,
            "token_pattern": None,
            "ngram_range": (1, 2),
            "min_df": 2,
            "sublinear_tf": True,
        },
    },
    "char_1_2": {
        "description": "字级 TF-IDF，1-2 gram",
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": (1, 2),
            "min_df": 2,
            "sublinear_tf": True,
        },
    },
    "char_1_3": {
        "description": "字级 TF-IDF，1-3 gram",
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": (1, 3),
            "min_df": 2,
            "sublinear_tf": True,
        },
    },
    "char_2_4": {
        "description": "字级 TF-IDF，2-4 gram",
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": (2, 4),
            "min_df": 2,
            "sublinear_tf": True,
        },
    },
    "char_2_3": {
        "description": "字级 TF-IDF，2-3 gram",
        "vectorizer": {
            "analyzer": "char",
            "ngram_range": (2, 3),
            "min_df": 2,
            "sublinear_tf": True,
        },
    },
}


def read_dataset(path):
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
        raise ValueError(f"{path.name} 字段不正确")
    return rows


def save_confusion_matrix(matrix, output_path):
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["真实\\预测", *LABEL_ORDER])
        for label, row in zip(LABEL_ORDER, matrix.tolist()):
            writer.writerow([label, *row])
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(LABEL_ORDER)),
        yticks=np.arange(len(LABEL_ORDER)),
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        xlabel="预测类别",
        ylabel="真实类别",
        title=output_path.stem.replace("confusion_matrix_", "混淆矩阵 "),
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
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
    figure.savefig(output_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    all_details = {}

    for experiment_id, config in EXPERIMENTS.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(**config["vectorizer"])),
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
        pipeline.fit(train_texts, train_labels)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predictions = pipeline.predict(test_texts)
        predict_seconds = time.perf_counter() - start

        vectorizer = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["classifier"]
        feature_names = vectorizer.get_feature_names_out()
        top_features = {}
        for class_index, class_name in enumerate(classifier.classes_):
            top_indices = np.argsort(classifier.coef_[class_index])[-12:][::-1]
            top_features[class_name] = [
                {
                    "feature": feature_names[index],
                    "weight": float(classifier.coef_[class_index, index]),
                }
                for index in top_indices
            ]
        matrix = confusion_matrix(test_labels, predictions, labels=LABEL_ORDER)
        report = classification_report(
            test_labels,
            predictions,
            labels=LABEL_ORDER,
            output_dict=True,
            zero_division=0,
        )
        mistakes = [
            {
                "text": text,
                "true_label": true_label,
                "predicted_label": predicted_label,
            }
            for text, true_label, predicted_label in zip(test_texts, test_labels, predictions)
            if true_label != predicted_label
        ]
        result = {
            "experiment_id": experiment_id,
            "description": config["description"],
            "vectorizer": {
                key: list(value) if isinstance(value, tuple) else str(value) if callable(value) else value
                for key, value in config["vectorizer"].items()
            },
            "feature_count": len(vectorizer.get_feature_names_out()),
            "accuracy": accuracy_score(test_labels, predictions),
            "macro_f1": f1_score(test_labels, predictions, average="macro", zero_division=0),
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "top_features": top_features,
            "mistakes": mistakes,
        }
        all_details[experiment_id] = result
        summary.append(
            {
                "experiment_id": experiment_id,
                "description": config["description"],
                "feature_count": result["feature_count"],
                "accuracy": round(result["accuracy"], 6),
                "macro_f1": round(result["macro_f1"], 6),
                "mistake_count": len(mistakes),
                "train_seconds": round(train_seconds, 6),
                "predict_seconds": round(predict_seconds, 6),
            }
        )
        save_confusion_matrix(matrix, RESULTS_DIR / f"confusion_matrix_{experiment_id}.csv")

    summary.sort(key=lambda row: (-row["macro_f1"], -row["accuracy"], row["feature_count"]))
    with (RESULTS_DIR / "experiment_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (RESULTS_DIR / "experiment_details.json").write_text(
        json.dumps(all_details, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "python_version": sys.version.split()[0],
        "word_tokenizer": WORD_TOKENIZER_NAME,
        "note": "词级实验使用 jieba.lcut；字级实验不依赖分词器。",
    }
    (RESULTS_DIR / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"ranking": summary, **metadata}, ensure_ascii=True))


if __name__ == "__main__":
    main()

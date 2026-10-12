"""Train the development label model and evaluate it on the recorded test set.

Run from the repository root with: python -m src.train
Only train_250.csv is used by fit(); test_50.csv is used by predict().
"""

from __future__ import annotations

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
from sklearn.pipeline import Pipeline

# 统一使用 label 命名；路径以本文件为基准，与终端当前目录无关。
ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "project" / "train_250.csv"
TRAIN_CSV2 = ROOT / "data" / "project" / "targeted_label_candidates_30(1).csv"
TRAIN_CSV3=ROOT/"data"/"project"/"error_data_diverse_30_complete(1).csv"
TRAIN_CSV4=ROOT/"data"/"project"/"targeted_label_candidates_30(1).csv"
TEST_CSV = ROOT / "data" / "project" / "test_50.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "label_model_dev.joblib"
INFO_PATH = MODEL_DIR / "model_info_dev.json"
RESULT_PATH = ROOT / "model_result" / "member_a_test_results.csv"
LABELS = ("宿舍设施", "校园网络", "食堂餐饮", "教学设施", "校园安全", "其他")
REQUIRED_COLUMNS = ("text", "label", "urgency", "department")


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
    frame1 = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    frame2 = pd.read_csv(TRAIN_CSV2, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    frame3 = pd.read_csv(TRAIN_CSV3, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    frame4 = pd.read_csv(TRAIN_CSV4, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    frame=pd.concat([frame1,frame2,frame3,frame4],ignore_index=True)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"训练文件缺少字段：{missing}")
    texts = frame["text"].str.strip()
    labels = frame["label"].str.strip()
    
    return texts, labels

#获取测试数据
def load_test_data() -> tuple[pd.Series, pd.Series, str]:
    if not TEST_CSV.is_file():
        raise FileNotFoundError(f"找不到项目测试数据：{TEST_CSV}")

    
    frame_test = pd.read_csv(TEST_CSV, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing_test = [column for column in REQUIRED_COLUMNS if column not in frame_test.columns]
    if missing_test:
        raise ValueError(f"测试文件缺少字段：{missing_test}")

    texts_test = frame_test["text"].str.strip()
    labels_test = frame_test["label"].str.strip()
    empty_rows_test = frame_test.index[texts_test.eq("")].tolist()
    invalid_rows_test = frame_test.index[~labels_test.isin(LABELS)].tolist()
    duplicate_rows_test = frame_test.index[texts_test.ne("") & texts_test.duplicated(keep=False)].tolist()
    if empty_rows_test or invalid_rows_test or duplicate_rows_test:
        raise ValueError(
            "测试数据需要先复核："
            f"空文本索引={empty_rows_test}，无效类别索引={invalid_rows_test}，"
            f"完全重复文本索引={duplicate_rows_test}"
        )
    return texts_test, labels_test,

def main() -> None:
    train_texts, train_labels = load_training_data()
    test_texts, test_labels= load_test_data()

    label_model = make_model()
    label_model.fit(train_texts, train_labels)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_model, MODEL_PATH)

    predictions = label_model.predict(test_texts)
    correct = int((predictions == test_labels.to_numpy()).sum())
    test_accuracy = float(accuracy_score(test_labels, predictions))
    results = pd.DataFrame({
        "测试表索引": test_texts.index,
        "诉求": test_texts.to_numpy(),
        "真实类别": test_labels.to_numpy(),
        "预测类别": predictions,
    })
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig", lineterminator="\n")

    info = {
        "status": "开发版 label 模型；已在指定的 50 条测试数据上评估",
        "model_file": "models/label_model_dev.joblib",
        "output_field": "label",
        "training_source": "data/project/train_250.csv,error_data_diverse_30_complete(1).csv,low_urgency_candidates_30.csv",
        "candidate_training_rows": len(train_texts),
        "test_source": "data/project/test_50.csv",
        "test_result_path": "model_result/member_a_test_results.csv",
        "test_result": {
            "correct": correct,
            "total": len(test_texts),
            "accuracy": test_accuracy,
        },
        "score_note": "模型用全部训练数据拟合后在这份测试集上得到此结果；若之后据此调整方案，同一测试集不再是全新独立检验。",
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
    with INFO_PATH.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(info, ensure_ascii=False, indent=2) + "\n")

    print(f"训练数据：（{len(train_texts)} 条）")
    print(f"已用测试数据：{TEST_CSV}（{len(test_texts)} 条）")
    print(f"本次测试：{correct}/{len(test_texts)}，准确率 {test_accuracy:.1%}")
    print(f"已保存 label 模型：{MODEL_PATH}")
    print(f"逐条结果：{RESULT_PATH}")
    print(f"版本记录：{INFO_PATH}")


if __name__ == "__main__":
    main()

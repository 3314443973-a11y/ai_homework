import json
import sys
from importlib.metadata import version
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "complaint_label_classifier_char_2_3.joblib"
REQUIRED_PYTHON = (3, 11, 16)
REQUIRED_PACKAGES = {
    "scikit-learn": "1.6.1",
    "joblib": "1.5.3",
    "numpy": "2.0.2",
    "scipy": "1.13.1",
    "jieba": "0.42.1",
    "matplotlib": "3.9.4",
}
EXPECTED_LABELS = {
    "宿舍设施",
    "校园网络",
    "食堂餐饮",
    "教学设施",
    "校园安全",
    "其他",
}


def main():
    python_version = sys.version_info[:3]
    if python_version != REQUIRED_PYTHON:
        raise RuntimeError(
            f"需要Python {'.'.join(map(str, REQUIRED_PYTHON))}，"
            f"当前为{'.'.join(map(str, python_version))}"
        )

    installed = {name: version(name) for name in REQUIRED_PACKAGES}
    mismatches = {
        name: {"required": required, "installed": installed[name]}
        for name, required in REQUIRED_PACKAGES.items()
        if installed[name] != required
    }
    if mismatches:
        raise RuntimeError(f"依赖版本不一致：{mismatches}")

    model = joblib.load(MODEL_PATH)
    classes = set(model.named_steps["classifier"].classes_)
    if classes != EXPECTED_LABELS:
        raise RuntimeError(f"模型类别不正确：{sorted(classes)}")

    result = {
        "status": "ok",
        "python_version": ".".join(map(str, python_version)),
        "packages": installed,
        "model_file": MODEL_PATH.name,
        "model_output": "label",
        "classes": sorted(classes),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

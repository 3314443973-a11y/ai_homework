
from pathlib import Path
import pandas as pd

from src.summary import summarize, local_summarize


# 项目路径
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "project" / "train_250.csv"
OUTPUT_PATH = ROOT /"summary"/ "summary_evaluation_results.csv"

# 自动尝试常见编码
def read_csv_safely(path):
    last_error = None

    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    raise last_error


def main():
    if not DATA_PATH.exists():
        print("找不到数据文件：", DATA_PATH)
        return

    df = read_csv_safely(DATA_PATH)

    if "text" not in df.columns:
        print("数据中没有 text 列，当前列名：", df.columns.tolist())
        return

    results = []

    for index, row in df.iterrows():
        original = row["text"]

        if not isinstance(original, str):
            original = ""

        summary = local_summarize(original)

        # 计算基本指标
        original_len = len(original.strip())
        summary_len = len(summary.strip())

        if original_len > 0:
            compression_ratio = round(
                1 - summary_len / original_len, 3
            )
        else:
            compression_ratio = 0

        results.append({
            "id": index + 1,
            "label": row.get("label", ""),
            "original_text": original,
            "summary": summary,
            "original_length": original_len,
            "summary_length": summary_len,
            "compression_ratio": compression_ratio,
            "summary_empty": not bool(summary.strip()),
            "summary_unchanged": (
                original.strip() == summary.strip()
            ),
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("=" * 50)
    print("摘要批量评测完成")
    print("=" * 50)
    print("原始诉求数量：", len(result_df))
    print("空摘要数量：", result_df["summary_empty"].sum())
    print("摘要与原文完全相同的数量：",
          result_df["summary_unchanged"].sum())
    print("平均原文长度：",
          round(result_df["original_length"].mean(), 2))
    print("平均摘要长度：",
          round(result_df["summary_length"].mean(), 2))
    print("平均压缩率：",
          round(result_df["compression_ratio"].mean() * 100, 2), "%")

    if "label" in result_df.columns:
        print("\n各类别测试数量：")
        print(result_df["label"].value_counts().to_string())

    print("\n结果文件已保存到：", OUTPUT_PATH)


if __name__ == "__main__":
    main()

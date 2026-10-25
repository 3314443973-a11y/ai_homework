
from collections import Counter
import re

import jieba
import pandas as pd

from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "project" / "train_250.csv"

print("数据文件路径：", DATA_PATH)
print("文件是否存在：", DATA_PATH.exists())

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
print("数据读取成功！")
print("数据条数：", len(df))
print("数据列名：", df.columns.tolist())

if "text" not in df.columns:
    raise ValueError("数据中没有 text 列，请检查 CSV 的列名。")

texts = df["text"].dropna().astype(str)
texts = texts[texts.str.strip() != ""]

print("有效诉求数量：", len(texts))
print("\n===== 1. 真实诉求样例 =====")

for i, text in enumerate(texts.head(10), start=1):
    print(f"{i}. {text}")

# 统计分词后的高频词
stopwords = {
    "希望", "请", "尽快", "一下", "我们", "他们",
    "这个", "那个", "已经", "还是", "就是", "感觉",
    "问题", "情况", "学校", "同学"
}

word_counter = Counter()
phrase_counter = Counter()
phrase_examples = {}

for text in texts:
    # 清理空白和常见标点
    text = re.sub(r"\s+", "", text)
    words = [
        w.strip()
        for w in jieba.lcut(text)
        if w.strip() and w.strip() not in stopwords
        and re.search(r"[\u4e00-\u9fff]", w)
    ]

    word_counter.update(words)

    # 统计相邻的两个词和三个词组成的短语
    for n in (2, 3):
        for i in range(len(words) - n + 1):
            phrase = "".join(words[i:i + n])

            # 排除过短或明显无意义的组合
            if len(phrase) < 3:
                continue

            phrase_counter[phrase] += 1
            phrase_examples.setdefault(phrase, text)

print("\n===== 2. 高频词 Top 30 =====")
for word, count in word_counter.most_common(30):
    print(f"{word}\t{count}")

print("\n===== 3. 高频短语 Top 40 =====")
for phrase, count in phrase_counter.most_common(40):
    print(f"{phrase}\t{count}\t例句：{phrase_examples[phrase]}")

# 保存统计结果，方便后续筛选
pd.DataFrame(
    word_counter.most_common(),
    columns=["word", "count"]
).to_csv(
    "summary_word_frequency.csv",
    index=False,
    encoding="utf-8-sig"
)

pd.DataFrame(
    [
        {
            "phrase": phrase,
            "count": count,
            "example": phrase_examples[phrase]
        }
        for phrase, count in phrase_counter.most_common()
    ]
).to_csv(
    "summary_phrase_frequency.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n统计完成，已生成：")
print("summary_word_frequency.csv")
print("summary_phrase_frequency.csv")

import pandas as pd

# 读取训练集和测试集
df_train = pd.read_csv("train_250.csv")
df_test = pd.read_csv("test_50.csv")
print("==训练集基本信息==")
print("训练集各个类别的数量：")
print(df_train["label"].value_counts())
print("==测试集基本信息==")
print("测试集各个类别的数量：")
print(df_test["label"].value_counts())
# 查看前十条样本
print("==查看前十条样本== ")
print("训练集前十条样本：")
print(df_train.head(10))
# 检查模糊样本
print("==检查模糊样本==")
for idx, row in df_train.iterrows():
    print(f"样本索引：{idx}，样本内容：{row['text']}，标签：{row['label']}")

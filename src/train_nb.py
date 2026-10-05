import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# 1.读取训练csv与测试csv
df_train = pd.read_csv("train_250.csv")
X_raw1 = df_train["text"]
y_train = df_train["label"]

df_test = pd.read_csv("test_50.csv")
X_raw2 = df_test["text"]
y_test = df_test["label"]

# 2.TF-IDF向量化
tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
# fit_transform将文本数据转换为TF-IDF特征矩阵
X_train = tfidf.fit_transform(X_raw1)
X_test = tfidf.transform(X_raw2)

# 3.初始化朴素贝叶斯模型
model = MultinomialNB()
# fit:模型学习训练数据
model.fit(X_train, y_train)

# 4.test数据预测
preds = model.predict(X_test)


# 输出预测结果
print("=====新诉求预测结果=====")
for txt, pred in zip(X_raw2, preds):
    print(f"文本: {txt}")
    print(f"预测标签: {pred}")

# 5.计算准确率
acc = accuracy_score(y_test, preds)
print(f"测试集准确率Accuracy = {acc:.4f}")

# 打印详细分类报告：各类准确率，召回率，F1值
print(classification_report(y_test, preds))

# 6.筛选误判样本
# 组装Dataframe
df_result = df_test.copy()
df_result["predicted_label"] = preds
df_mistakes = df_result[df_result["label"] != df_result["predicted_label"]]
print("=====误判样本=====")
for index, row in df_mistakes.iterrows():
    print(f"文本: {row['text']}")
    print(f"真实标签: {row['label']}, 预测标签: {row['predicted_label']}")


# 7.绘制混淆矩阵
cm = confusion_matrix(y_test, preds)
print("=====混淆矩阵=====")
print(cm)

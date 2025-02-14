import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import plotly.express as px
import numpy as np

# 设置支持中文的字体，SimHei（黑体）
plt.rcParams['font.family'] = 'SimHei'
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 读取训练数据
data = pd.read_csv("TrainSet.csv")
x = data['text_raw']
y = data['sentiment']

# 特征提取：使用词袋模型将文本转换为数值特征
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(x)

# 应用 SMOTE 过采样算法
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 模型训练：使用多项式朴素贝叶斯分类器
model = MultinomialNB()
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy}")

# 读取新文本数据
new_texts = pd.read_csv("./../crawler/review.csv")

# 对新文本进行特征转换
new_X = vectorizer.transform(new_texts['text_raw'])

# 进行情感分类
new_predictions = model.predict(new_X)

# 将预测结果添加到 new_texts 数据框中
new_texts['sentiment'] = new_predictions

# 统计各类情感的文本数量
sentiment_counts = pd.Series(new_predictions).value_counts()
print("各类情感的文本数量统计：")
print(sentiment_counts)

# 创建一个 2 行 2 列的子图布局
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 饼图：展示积极、消极、中性情感文本占比情况
labels = sentiment_counts.index
sizes = sentiment_counts.values
axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
axes[0, 0].axis('equal')
axes[0, 0].set_title('不同情感文本占比情况')

# 2. 柱状图：对比不同情感文本的数量
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=axes[0, 1])
axes[0, 1].set_xlabel('情感类别')
axes[0, 1].set_ylabel('文本数量')
axes[0, 1].set_title('不同情感文本的数量对比')

# 生成词云图
positive_texts = ' '.join(new_texts[new_predictions == '积极']['text_raw'])
words = positive_texts.split()
# 定义需要过滤的词汇模式，使用正则表达式
filter_patterns = ["回复.*", "关注.*"]
# 过滤掉符合模式的词汇
filtered_words = []
for word in words:
    valid = True
    for pattern in filter_patterns:
        if re.search(pattern, word):
            valid = False
            break
    if valid:
        filtered_words.append(word)
top_20_words = [word for word, _ in Counter(filtered_words).most_common(20)]
top_20_text = ' '.join(top_20_words)
# 设微软雅黑字体路径
font_path = r'C:\Windows\Fonts\msyh.ttc'
wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(top_20_text)
axes[1, 0].imshow(wordcloud, interpolation='bilinear')
axes[1, 0].axis('off')
axes[1, 0].set_title('积极情感文本中出现频率较高的前 20 个词汇词云')

# 4. 折线图：展示随着时间推移，不同情感倾向的文本数量变化趋势（以每天为时间间隔进行统计）
if 'created_at' in new_texts.columns:
    # 指定日期时间格式进行转换
    new_texts['created_at'] = pd.to_datetime(new_texts['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    new_texts['date'] = new_texts['created_at'].dt.date
    daily_counts = new_texts.groupby(['date', 'sentiment']).size().unstack()
    for sentiment in daily_counts.columns:
        axes[1, 1].plot(daily_counts.index.astype(str), daily_counts[sentiment], label=sentiment)
    axes[1, 1].set_xlabel('日期')
    axes[1, 1].set_ylabel('文本数量')
    axes[1, 1].set_title('不同情感倾向的文本数量随日期变化趋势')
    axes[1, 1].legend()
else:
    axes[1, 1].text(0.5, 0.5, "新文本数据中缺少 'created_at' 列，无法绘制折线图。",
                    horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()
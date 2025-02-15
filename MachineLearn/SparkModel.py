from pyspark.sql import SparkSession
from pyspark.ml.feature import CountVectorizer, StringIndexer, Tokenizer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, explode, array, lit

# 创建SparkSession
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .getOrCreate()

# 从HDFS读取训练数据
train_data_path = "hdfs:///2307020220/TrainSet.csv"
data = spark.read.csv(train_data_path, header=True, inferSchema=True)

# 统计各类别的样本数量
class_counts = data.groupBy("sentiment").count().collect()
class_counts_dict = {row["sentiment"]: row["count"] for row in class_counts}

# 找出最大的类别样本数量
max_count = max(class_counts_dict.values())

# 随机过采样函数
def random_oversample(df, class_col, max_count):
    temp_df = df
    for class_label, count in class_counts_dict.items():
        if count < max_count:
            ratio = int(max_count / count)
            oversampled_df = df.filter(col(class_col) == class_label).withColumn("dummy", explode(array([lit(1)] * ratio))).drop("dummy")
            temp_df = temp_df.union(oversampled_df)
    return temp_df

# 进行过采样
oversampled_data = random_oversample(data, "sentiment", max_count)

# 分词处理：将 text_raw 列从 string 类型转换为 array<string> 类型
tokenizer = Tokenizer(inputCol="text_raw", outputCol="text_tokens")

# 特征提取：使用词袋模型将文本转换为数值特征
vectorizer = CountVectorizer(inputCol="text_tokens", outputCol="features")
indexer = StringIndexer(inputCol="sentiment", outputCol="label")

# 划分训练集和测试集
(train_data, test_data) = oversampled_data.randomSplit([0.8, 0.2], seed=42)

# 模型训练：使用多项式朴素贝叶斯分类器
nb = NaiveBayes(featuresCol="features", labelCol="label")

# 创建Pipeline
pipeline = Pipeline(stages=[tokenizer, vectorizer, indexer, nb])

# 训练模型
model = pipeline.fit(train_data)

# 模型评估
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"模型准确率: {accuracy}")

# 从HDFS读取新文本数据
new_texts_path = "hdfs:///2307020220/review.csv"
new_texts = spark.read.csv(new_texts_path, header=True, inferSchema=True)

# 进行情感分类
new_predictions = model.transform(new_texts)

# 将预测结果转换为Pandas DataFrame
new_predictions_pd = new_predictions.toPandas()

# 统计各类情感的文本数量
sentiment_counts = new_predictions_pd['prediction'].value_counts()
print("各类情感的文本数量统计：")
print(sentiment_counts)

# 停止SparkSession
spark.stop()
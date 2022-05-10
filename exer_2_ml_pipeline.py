from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, IDF, StopWordsRemover, CountVectorizer, ChiSqSelector, HashingTF
from pyspark.ml.feature import StringIndexer, Normalizer
from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


REVIEW_INPUT_DATA_PATH = "./reviews_devset.json"
STOP_WORDS_PATH = "./stopwords.txt"

spark = SparkSession.builder.appName('spark_ml_pipeline').getOrCreate()
df = spark.read.format("json").load(REVIEW_INPUT_DATA_PATH)

(train, test) = df.randomSplit([0.1, 0.1], seed=42)
df= train

stop_words = []
with open(STOP_WORDS_PATH, encoding='utf-8') as f:
    stop_words = [line.rstrip() for line in f.readlines()]

df_select = df.select("category", "reviewText")

(train, test) = df_select.randomSplit([0.8, 0.2], seed=42)

tokenizer = RegexTokenizer(inputCol="reviewText", outputCol="words", pattern="[0-9\(\)\[\]\{\}\.\!\?\,\;\:\+\=\-\_\"\'\`~#@&*%€$§\\\/\t ]", minTokenLength=1)

str_indexer = StringIndexer(inputCol="category", outputCol="label")

remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=stop_words)

count_vectorizer = CountVectorizer(inputCol="filtered", outputCol="rawFeatures")

idf = IDF(inputCol="rawFeatures", outputCol="features")

selector = ChiSqSelector(numTopFeatures=2000, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="label")

normalizer =  Normalizer(inputCol="selectedFeatures", outputCol="normFeatures")

svc = LinearSVC(maxIter=10, regParam=0.1)

one_vs_rest = OneVsRest( classifier=svc)

pipeline = Pipeline(stages=[tokenizer,str_indexer, remover, count_vectorizer, idf, selector, normalizer, one_vs_rest])

param_grid = ParamGridBuilder() \
    .addGrid(svc.maxIter, [5,15])    \
    .addGrid(svc.regParam, [0.1, 0.4, 0.7]) \
    .addGrid(svc.standardization, [True, False]) \
    .build()

# paramGrid = ParamGridBuilder()
# paramGrid.addGrid(svc.maxIter, [1, 10])
# paramGrid.addGrid(svc.regParam, [0.1, 0.4])
# paramGrid.addGrid(svc.standardization, [True, False])
# paramGrid.build()
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=4
                          )

model = crossval.fit(train)
predictions = model.transform(test)

evaluator.evaluate(predictions)
# Databricks notebook source
import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType, IntegerType

dataFile = "dbfs:///FileStore/tables/framingham.csv"

dataDF = spark.read.format("csv").option("header", True).option("inferSchema",True).option("ignoreLeadingWhiteSpace",True).option("mode", "dropMalformed").load(dataFile)


dataDF = dataDF.select(dataDF['male'], dataDF['age'], dataDF['currentSmoker'], dataDF['diabetes'], dataDF['heartRate'], dataDF['TenYearCHD'])

dataDF = dataDF.withColumn("heartRate", f.col("heartRate").astype(IntegerType()))

dataDF = dataDF.withColumnRenamed('TenYearCHD','label')

rawdataDF = dataDF.na.drop()

rawdataDF.show()

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, Bucketizer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression

# Building the Pipeline

# Age broken into categories: below 35, 35-44, 45-54, 55-64, 65 and above.
ageSplits =  [-float("inf"), 35, 45, 55, 65, float("inf")]
ageBucketizer = Bucketizer(splits=ageSplits, inputCol="age", outputCol="ageBucket")


heartrateSplits =  [-float("inf"), 60, 90, float("inf")]
heartrateBucketizer = Bucketizer(splits=heartrateSplits, inputCol="heartRate", outputCol="heartRateBucket")


vecAssem = VectorAssembler(inputCols=['ageBucket','heartRateBucket','male','currentSmoker','diabetes'],outputCol='features')


lr = LogisticRegression(maxIter=20, regParam = 0.01)

myStages=[ageBucketizer, heartrateBucketizer, vecAssem, lr]
p = Pipeline(stages=myStages)

trainDF, testDF = rawdataDF.randomSplit([0.8, 0.2], seed = 123)

# fit the pipeline on the training data
pModel = p.fit(trainDF)

# COMMAND ----------

# test the pipeline model on testing data

pred_train = pModel.transform(trainDF)
pred_train.select('label', 'probability','prediction').show()

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print(evaluator.evaluate(pred_train))

# COMMAND ----------

pred_test = pModel.transform(testDF)
pred_test.select('label', 'probability','prediction').show()

evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction")
print(evaluator.evaluate(pred_test))

# COMMAND ----------

dbutils.fs.rm("FileStore/tables/as2", True)
testDF.write.format("csv").option("header", True).save("FileStore/tables/as2/")

# COMMAND ----------

sourceStream = spark.readStream.format("csv").option("header", True).option("maxFilePerTrigger", 1).schema(dataSchema).load("dbfs:///FileStore/tables/as2")

sourceWin = sourceStream.groupBy("male").agg(f.count("male").alias("count"))

sinkStream = hashtagsWin.writeStream.outputMode("complete").format("memory").queryName("sourceWin").start()

# COMMAND ----------



from __future__ import print_function
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
if __name__ == "__main__":

    # 
    spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/temp").appName("LinearRegression").getOrCreate()

    # convert data to MLLib format 
    inputdata = spark.sparkContext.textFile("regression.txt")
    data = inputdata.map(lambda x: x.split(",")).map(lambda x: (float(x[0]), Vectors.dense(float(x[1]))))

    # Convert this RDD to a DataFrame
    ColumnLable = ["label", "features"]
    df = data.toDF(ColumnLable)

    # split data 
    TrainTestDF = df.randomSplit([0.5, 0.5])
    DataTrain = TrainTestDF[0]
    testDF = TrainTestDF[1]

    # linear regression model
    ModelLinearRegression = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
    model = ModelLinearRegression.fit(DataTrain)
    #Prediction
    LinearPred = model.transform(testDF).cache()
    predictions = LinearPred.select("prediction").rdd.map(lambda x: x[0])
    labels = LinearPred.select("label").rdd.map(lambda x: x[0])
    predictionAndLabel = predictions.zip(labels).collect()

    # output
    for labelsP in predictionAndLabel:
      print(labelsP)

    # Stop the session
    spark.stop()

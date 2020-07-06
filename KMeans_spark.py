from pyspark.mllib.clustering import KMeans
from numpy import array, random
from math import sqrt
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import scale

K = 5
conf = SparkConf().setMaster("local").setAppName("SparkKMeans")
sc = SparkContext(conf = conf)

#Create clusters for N people in k clusters
def createDFKmean(N, k):
    random.seed(123)
    numberEach = float(N)/k
    X = []
    for i in range (k):
        varFirstCenter = random.uniform(20000.0, 200000.0)
        varSecondCenter = random.uniform(20.0, 70.0)
        for j in range(int(numberEach)):
            X.append([random.normal(varFirstCenter, 10000.0), random.normal(varSecondCenter, 2.0)])
    X = array(X)
    return X

random.seed(123)

# normalize
data = sc.parallelize(scale(createDFKmean(100, K)))

# Model Fit
clusters = KMeans.train(data, K, maxIterations=10,
        initializationMode="random")
KmeanResult = data.map(lambda point: clusters.predict(point)).cache()

print()
print("Counts by value:")
print(KmeanResult.countByValue())

print("Cluster assigned:")
print(KmeanResult.collect())


# Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = data.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))

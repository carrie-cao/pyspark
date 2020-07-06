from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array
conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf = conf)
# convertCSV input into numerical
def NumYN(YN):
    if (YN == 'Y'):
        return 1
    else:
        return 0

def EducationLevel(degree):
    if (degree == 'BS'):
        return 1
    elif (degree =='MS'):
        return 2
    elif (degree == 'PhD'):
        return 3
    else:
        return 0

# Convert LabeledPoint for MLLib
def lablepointForMLLib(fields):
    WorkYears = int(fields[0])
    EmployedYN = NumYN(fields[1])
    EmployerEx = int(fields[2])
    educationLevel = EducationLevel(fields[3])
    LevelHIgh = NumYN(fields[4])
    Internship = NumYN(fields[5])
    HireYN = NumYN(fields[6])

    return LabeledPoint(HireYN, array([WorkYears, EmployedYN,
        EmployerEx, educationLevel, LevelHIgh, Internship]))

#Load data
PastHdf = sc.textFile("PastHires.csv")
header = PastHdf.first()
PastHdf = PastHdf.filter(lambda x:x != header)
dfToCsv = PastHdf.map(lambda x: x.split(","))

# Convert to LabeledPoints
trainingData = dfToCsv.map(lablepointForMLLib)

# Create a test individual
dfTest = [ array([10, 1, 3, 1, 0, 0])]
TestDf = sc.parallelize(dfTest)

# Train model
DecisionTreeModel = DecisionTree.trainClassifier(trainingData, numClasses=2,
                                     categoricalFeaturesInfo={1:2, 3:4, 4:2, 5:2},
                                     impurity='gini', maxDepth=5, maxBins=32)

# test model
predictions = DecisionTreeModel.predict(TestDf)
print('Hire prediction:')
results = predictions.collect()
for result in results:
    print(result)

# print out the decision tree itself:
print('Learned classification tree DecisionTreeModel:')
print(DecisionTreeModel.toDebugString())

from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
conf = SparkConf().setMaster("local").setAppName("SparkTFIDF")
sc = SparkContext(conf = conf)

# data
DataOrig = sc.textFile("subset-small.tsv")
InputRecord = DataOrig.map(lambda x: x.split("\t"))
FilesRecord = InputRecord.map(lambda x: x[3].split(" "))
FileLables = InputRecord.map(lambda x: x[1])

# hash the words to their term frequencies:
hashingTF = HashingTF(1000000)  #hash buckets 
tf = hashingTF.transform(FilesRecord)
# compute TF*IDF
tf.cache()
idf = IDF(minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

# check "Gettysburg" 
gettysburgTF = hashingTF.transform(["Gettysburg"])
gettysburgHashValue = int(gettysburgTF.indices[0])

# extract the TF*IDF score
gettysburgRelevance = tfidf.map(lambda x: x[gettysburgHashValue])
zippedResults = gettysburgRelevance.zip(FileLables)

# maximum TF*IDF value:
print()
print("maximum TF*IDF value for Gettysburg is:")
print(zippedResults.max())

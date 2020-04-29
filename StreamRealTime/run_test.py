from pyspark import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext()
spark = SparkSession.builder.getOrCreate()

spkDF = spark.read.csv('/Users/fer/anaconda/pkgs/blaze-0.10.1-py36_0/lib/python3.6/site-packages'
                       '/blaze/examples/data/accounts_1.csv')
spark.show()

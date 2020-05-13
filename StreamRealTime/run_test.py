import pyspark

from pyspark import SparkContext
from pyspark.sql.session import SparkSession
import os

# os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Content'
# os.environ['JAVA_HOME'] = 'opt/jdk/jdk1.8.0_151/'
# os.getenv('JAVA_HOME')

sc = SparkContext()
spark = SparkSession.builder.getOrCreate()
print('Spark version: ', spark.version)
print('Context version: ', sc.version)
print('Dataframes on RAM Memory: ', spark.catalog.listTables())


spkDF = spark.read.csv('/Users/fer/anaconda/pkgs/blaze-0.10.1-py36_0/lib/python3.6/site-packages'
                       '/blaze/examples/data/accounts_1.csv')
spark.show()

df = spark.read.format('csv').option('header', 'true')\
    .option('inferSchema', 'false').load('/Users/fer/Desktop/Escritorio - MacBook Pro de Fernando/BlockChain_Train_csv.csv')

print(sc.version)
print(spark.version)
print(spark.catalog.listTables())

df2 = df[['Date', 'Close', 'USD_Exchange_Trade_Volume', 'Bitcoins_in_circulation', 'MarketCap', 'BlockSize',
          'USD/AUD', 'USD/EUR', 'Difficulty']]
df2.show()

len(df2.columns) # number of columns
df2.count() # nº of rows
df2.createOrReplaceTempView('df2_RAM')

df2.select('Difficulty').distinct() # unique values

# Bitcoins in circulation per Difficulty level
bitcoins_per_difficulty = spark.sql(''' 
select Difficulty, AVG(Bitcoins_in_circulation) as mean
from df2_RAM
group by Difficulty
 ''')

import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql.types import *
df2 = df2.withColumn('Bitcoins_in_circulation', df2['Bitcoins_in_circulation'].cast(IntegerType()))
x = df2.groupBy('Difficulty').mean('Bitcoins_in_circulation')

bitcoins_per_difficulty_df = bitcoins_per_difficulty.toPandas()
print(bitcoins_per_difficulty_df)
bitcoins_per_difficulty_df.to_csv('/Users/fer/Desktop/Escritorio - MacBook Pro de Fernando/test_spark_sql.csv')

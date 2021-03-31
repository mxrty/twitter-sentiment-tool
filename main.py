import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, IntegerType, LongType

if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("Twitter Sentiment Tool").getOrCreate()

    schema = StructType() \
        .add("polarity", IntegerType(), True) \
        .add("tweet_id", LongType(), True) \
        .add("date", StringType(), True) \
        .add("query", StringType(), True) \
        .add("user", StringType(), True) \
        .add("text", StringType(), True)

    # .schema(schema) \

    df = spark.read.format("csv") \
        .option("inferSchema", True) \
        .load("/home/m/CS3800/twitter-sentiment-tool/data/training.1600000.processed.noemoticon.csv") \
        .toDF("polarity","tweet_id","datetime","query","user","text")

    df2 = df.drop("query")

    df2.printSchema()

    df2.show(10)
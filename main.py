import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, DoubleType, BooleanType

if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("Twitter Sentiment Tool").getOrCreate()

    schema = StructType() \
        .add("Polarity", StringType(), True) \
        .add("Tweet ID", StringType(), True) \
        .add("Date", StringType(), True) \
        .add("Query", StringType(), True) \
        .add("User", StringType(), True) \
        .add("Text", StringType(), True)

    df = spark.read.format("csv") \
        .schema(schema) \
        .load("/home/m/CS3800/twitter-sentiment-tool/data/training.1600000.processed.noemoticon.csv")

    df.show(10)
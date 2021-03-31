import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp

if __name__ == "__main__":
    spark = SparkSession.builder.master("local[*]").appName("Twitter Sentiment Tool").getOrCreate()

    # Have to set legacy parsing as Spark 3.0+ cannot use 'E'
    spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

    df = spark.read.format("csv") \
        .option("inferSchema", True) \
        .load("/home/m/CS3800/twitter-sentiment-tool/data/training.1600000.processed.noemoticon.csv") \
        .toDF("polarity","tweet_id","datetime","query","user","text")

    df2 = df.withColumn('timestamp',to_timestamp("datetime", "EEE MMM dd HH:mm:ss zzz yyyy"))

    df3 = df2.drop("query").drop("datetime")

    df3.printSchema()

    df3.show(10)
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    to_timestamp,
    col,
    lower,
    regexp_replace,
    sum,
    explode,
)
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    DoubleType,
    BooleanType,
)

# Smaller file (500 rows) to use when testing
default_file_path = "data/testdata.manual.2009.06.14.csv"

# Full file (1.6M rows)
# default_file_path = "data/training.1600000.processed.noemoticon.csv"


spark = (
    SparkSession.builder.master("local[*]")
    .appName("Twitter Sentiment Tool")
    .getOrCreate()
)

tweet_schema = StructType(
    [
        StructField("word", StringType(), True),
        StructField("found", BooleanType(), True),
        StructField("avg_sentiment", DoubleType(), True),
        StructField("count", LongType(), True),
    ]
)


def clean_text(text):
    text = lower(text)
    text = regexp_replace(text, "^rt ", "")
    text = regexp_replace(text, "(https?\://)\S+", "")
    text = regexp_replace(text, "[^a-zA-Z0-9\\s]", "")
    text = regexp_replace(text, "\s\s+", " ")
    return text


# Sum all values in a column
def sum_col(df, col):
    return df.select(sum(col)).collect()[0][0]


# Create dataframe from csv
def init_base_df(file_path=default_file_path):
    # Set legacy parsing as Spark 3.0+ cannot use 'E' for timestamp
    spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

    print("Loading", default_file_path)

    raw_df = (
        spark.read.format("csv")
        .option("inferSchema", True)
        .load(file_path)
        .toDF("polarity", "tweet_id", "datetime", "query", "user", "text")
    )

    # Parse string to timestamp
    time_parsed_df = raw_df.withColumn(
        "timestamp", to_timestamp("datetime", "EEE MMM dd HH:mm:ss zzz yyyy")
    )

    df = time_parsed_df.drop("query").drop("datetime")

    # Shift polarity from a range of [0:4], to [-1:1]
    scaled_polarity_df = df.withColumn("sentiment", (col("polarity") / 2) - 1).drop(
        "polarity"
    )

    clean_text_df = df.select(clean_text(col("text")).alias("text"), "tweet_id")

    tokenizer = Tokenizer(inputCol="text", outputCol="vector")
    vector_df = tokenizer.transform(clean_text_df).select("vector", "tweet_id")

    remover = StopWordsRemover()
    stopwords = remover.getStopWords()

    remover.setInputCol("vector")
    remover.setOutputCol("tokens")

    tokens_no_stopw_df = remover.transform(vector_df).select("tokens", "tweet_id")

    tweets_with_tokens_df = scaled_polarity_df.join(tokens_no_stopw_df, on=["tweet_id"])

    return tweets_with_tokens_df


# Create dataframe with each words average sentiment value
def init_word_sentiments_df(base_df):
    words_exploded_df = base_df.select("sentiment", explode("tokens").alias("word"))

    counts_df = words_exploded_df.groupBy("word").count()

    sentiments_df = words_exploded_df.groupBy("word").agg(
        sum("sentiment").alias("total_sentiment")
    )

    return counts_df.join(sentiments_df, on=["word"]).withColumn(
        "avg_sentiment", col("total_sentiment") / col("count")
    )


# Create dataframe with each users average sentiment value
def init_user_sentiments_df(base_df):
    user_tweet_counts_df = base_df.select("user").groupBy("user").count()

    sentiments_df = base_df.groupBy("user").agg(
        sum("sentiment").alias("total_sentiment")
    )

    return user_tweet_counts_df.join(sentiments_df, on=["user"]).withColumn(
        "avg_sentiment", col("total_sentiment") / col("count")
    )


def init_tweet_row(word=None, found=None, avg_sentiment=None, count=None):
    row_data = []
    # If no method arguments create empty row (header)
    if (
        word is not None
        and found is not None
        and avg_sentiment is not None
        and count is not None
    ):
        row_tuple = (word, found, avg_sentiment, count)
        row_data.append(row_tuple)
    return spark.createDataFrame(row_data, tweet_schema)

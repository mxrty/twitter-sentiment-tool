import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    to_timestamp,
    col,
    lower,
    regexp_replace,
    hour,
    sum,
    explode,
)
from pyspark.ml.feature import Tokenizer, StopWordsRemover

# TODO: Swap clean and tokenise processes
def clean_text(text):
    text = lower(text)
    # html attributes
    # text = regexp_replace(text, "@[A-Za-z0-9_]+", "")
    text = regexp_replace(text, "^rt ", "")
    text = regexp_replace(text, "(https?\://)\S+", "")
    text = regexp_replace(text, "[^a-zA-Z0-9\\s]", "")
    text = regexp_replace(text, "\s\s+", " ")
    # text = text.strip()
    return text


def sum_col(df, col):
    return df.select(sum(col)).collect()[0][0]


def init_base_df():
    # file_path = "/home/m/CS3800/twitter-sentiment-tool/data/training.1600000.processed.noemoticon.csv"
    file_path = (
        "/home/m/CS3800/twitter-sentiment-tool/data/testdata.manual.2009.06.14.csv"
    )

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("Twitter Sentiment Tool")
        .getOrCreate()
    )

    # Set legacy parsing as Spark 3.0+ cannot use 'E' for timestamp
    spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

    df = (
        spark.read.format("csv")
        .option("inferSchema", True)
        .load(file_path)
        .toDF("polarity", "tweet_id", "datetime", "query", "user", "text")
    )

    # Parse string to timestamp
    df2 = df.withColumn(
        "timestamp", to_timestamp("datetime", "EEE MMM dd HH:mm:ss zzz yyyy")
    )

    df3 = df2.drop("query").drop("datetime")

    # Shift polarity from a range of [0:4], to [-1:1]
    scaled_polarity_df = df3.withColumn("sentiment", (col("polarity") / 2) - 1).drop(
        "polarity"
    )

    clean_text_df = df3.select(clean_text(col("text")).alias("text"), "tweet_id")

    tokenizer = Tokenizer(inputCol="text", outputCol="vector")
    vector_df = tokenizer.transform(clean_text_df).select("vector", "tweet_id")

    remover = StopWordsRemover()
    stopwords = remover.getStopWords()

    remover.setInputCol("vector")
    remover.setOutputCol("tokens")

    vector_no_stopw_df = remover.transform(vector_df).select("tokens", "tweet_id")

    tweets_with_tokens_df = scaled_polarity_df.join(vector_no_stopw_df, on=["tweet_id"])

    return tweets_with_tokens_df


# 1. In: hour, Out: sentiment shift at hour
def sentiment_at_hour(base_df, input_hour):
    tweets_at_hour_df = tweets.where(hour(col("timestamp")) == input_hour)
    print(
        f"Average sentiment bias from {input_hour}:00 to {input_hour}:59 : ",
        sum_col(tweets_at_hour_df, "sentiment") / tweets_at_hour_df.count(),
    )


def init_word_sentiments_df(base_df):
    # Create dataframe where each word has a sentiment value
    words_exploded_df = tweets.select("sentiment", explode("tokens").alias("word"))

    counts_df = words_exploded_df.groupBy("word").count()

    sentiments_df = words_exploded_df.groupBy("word").agg(
        sum("sentiment").alias("total_sentiment")
    )

    words_df = counts_df.join(sentiments_df, on=["word"])

    return words_df


if __name__ == "__main__":

    tweets = init_base_df()

    in_hour = 1
    sentiment_at_hour(tweets, in_hour)

    word_sentiments_df = init_word_sentiments_df(tweets)
    word_sentiments_df.show()

    # 3. In: sentiment, Out: words with sentiment
    # input_sentiment = "positive"
    # if(input_sentiment == "positive"):

    # elif(input_sentiment = "negative"):

    # elif(input_sentiment = "neutral"):

    # else:
    # Invalid input

    # 4. In: word, Out: sentiment
    # 6. In: sentiment, Out: users with sentiment
    # 7. In: sentence (word[]), Out: sentiment for each word

    # 2. In: text (140 char), Out: sentiment
    # 5. In: text (140 char), Out: similair tweets and their sentiments

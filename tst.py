import click
from jobs.init_dataframes import (
    init_base_df,
    init_word_sentiments_df,
    init_user_sentiments_df,
    init_tweet_row,
    sum_col,
)
import pyspark
from pyspark.sql.functions import col, hour as to_hour, lower


# Command Line Interface entry point
@click.group()
def cli():
    """
    spark-submit tst.py COMMAND [--ARGS] <INPUT>
    """
    print("Loading file...")
    global base_df
    base_df = init_base_df()
    pass


# Input: hour, Output: sentiment shift at hour
@cli.command()
@click.argument("hour", default=1, type=int)
def sentiment_at_hour(hour):
    tweets_at_hour_df = base_df.where(to_hour(col("timestamp")) == hour)
    avg_sentiment = sum_col(tweets_at_hour_df, "sentiment") / tweets_at_hour_df.count()
    print(f"Average sentiment bias from {hour}:00 to {hour}:59 : {avg_sentiment}")


# Input: word, Output: avg sentiment of word
@cli.command()
@click.argument("word", type=str)
def sentiment_of_word(word):
    word_sentiments_df = init_word_sentiments_df(base_df)
    rows = word_sentiments_df.where(word_sentiments_df.word == word).collect()
    if rows:
        row = rows[0]
        freq = row["count"]
        sentiment = row["avg_sentiment"]
        print(f"Average sentiment for '{word}' : ", sentiment, f" (samples = {freq})")
    else:
        print(f"Word '{word}'' not found in dataframe.")


# Input: user, Output: avg sentiment of user's tweets
@cli.command()
@click.argument("user", type=str)
def sentiment_of_user(user):
    user_sentiments_df = init_user_sentiments_df(base_df)
    rows = user_sentiments_df.where(user_sentiments_df.user == user).collect()
    if rows:
        row = rows[0]
        freq = row["count"]
        sentiment = row["avg_sentiment"] / freq
        print(f"Average sentiment for '{user}' : ", sentiment, f" (samples = {freq})")
    else:
        print(f"User '{user}' not found in dataframe.")


# Input: sentiment, min occurrences, max words to return, Output: words with specified sentiment
@cli.command()
@click.option("--positive/--negative", default=True)
@click.option("--min-samples", type=int, default=2)
@click.option("--max", type=int, default=20)
def words_by_sentiment(positive, min_samples, max):
    word_sentiments_df = init_word_sentiments_df(base_df)
    filtered_word_sentiments_df = word_sentiments_df.where(col("count") >= min_samples)

    filtered_word_sentiments_df.orderBy(
        ["avg_sentiment", "count", "word"], ascending=not positive
    ).show(max)


# Input: sentiment, min occurrences, max users to return, Output: users with specified sentiment
@cli.command()
@click.option("--positive/--negative", default=True)
@click.option("--min-samples", type=int, default=2)
@click.option("--max", type=int, default=20)
def users_by_sentiment(positive, min_samples, max):
    user_sentiments_df = init_user_sentiments_df(base_df)
    filtered_user_sentiments_df = user_sentiments_df.where(col("count") >= min_samples)

    filtered_user_sentiments_df.orderBy(
        ["avg_sentiment", "count", "user"], ascending=not positive
    ).show(max)


# Input: sentence (word[]), Output: avg sentiment for each word
@cli.command()
@click.argument("tweet", type=str)
def sentiment_of_tweet(tweet):
    if len(tweet) > 140:
        print("Tweets must be a maximum of 140 characters.")
    else:
        print(f"Tweet: '{tweet}'")
        word_sentiments_df = init_word_sentiments_df(base_df)
        tweet_sentiment_df = init_tweet_row()

        tweet_words = tweet.split()
        for word in tweet_words:
            found = False
            avg_sentiment = 0.0
            count = 0

            rows = word_sentiments_df.where(word_sentiments_df.word == word).collect()
            if rows:
                row = rows[0]
                found = True
                avg_sentiment = row["avg_sentiment"]
                count = row["count"]

            new_row = init_tweet_row(word, found, avg_sentiment, count)
            tweet_sentiment_df = tweet_sentiment_df.union(new_row)

        tweet_sentiment_df.show()


if __name__ == "__main__":
    cli()

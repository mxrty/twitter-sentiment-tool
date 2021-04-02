import click
from jobs.init_dfs import init_base_df, init_word_sentiments_df, init_user_sentiments_df
import pyspark
from pyspark.sql.functions import col


# Command Line Interface entry point
@click.group()
def cli():
    """ Usage info."""
    print("Loading file...")
    global base_df
    base_df = init_base_df()
    pass


# Input: hour, Output: sentiment shift at hour
@cli.command()
@click.argument("hour", default=1, type=int)
def sentiment_at_hour(hour):
    tweets_at_hour_df = base_df.where(to_hour(col("timestamp")) == hour)
    print(
        f"Average sentiment bias from {hour}:00 to {hour}:59 : ",
        sum_col(tweets_at_hour_df, "sentiment") / tweets_at_hour_df.count(),
    )


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
        ["avg_sentiment", "word"], ascending=not positive
    ).show(max)


if __name__ == "__main__":
    cli()
    # 6. In: sentiment, Out: users with sentiment
    # 7. In: sentence (word[]), Out: sentiment for each word

    # 2. In: text (140 char), Out: sentiment
    # 5. In: text (140 char), Out: similar tweets and their sentiments

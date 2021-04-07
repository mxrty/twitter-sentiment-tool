# twitter-sentiment-tool
A project built using Apache Spark's python library - PySpark\
Uses the [Sentiment140 dataset](http://help.sentiment140.com/for-students/)\
Can be operated using CLI built with Click

**Developed and tested on Linux (Ubuntu 20.04)**

## Installation
### Prerequisites:
These need to be installed on your machine (and create system PATH variables):
* Apache Spark 3.1.1 built for Hadoop 3.2+ [here](https://spark.apache.org/downloads.html)
* Java
* Python3

Then, [install pip](https://linuxize.com/post/how-to-install-pip-on-ubuntu-20.04/)

### Steps:

1\. Install pipenv:

#### `pip3 install pipenv`

or 

#### `pip install pipenv`

2\. Clone repository:

#### `git clone https://github.com/mxrty/twitter-sentiment-tool.git`

3\. Install packages:

#### `pipenv install`

4\. Run virtual-env shell:

#### `pipenv shell`

5\. Set up full Sentiment140 dataset *(Optional)* :

*This repository contains a 500 line file which will work by default, but does not contain the full dataset as this is over GitHub's 100MB limit.*

Download Sentiment140 dataset and unzip into the `/data` folder. Then go into `/jobs/init_dataframes.py` and uncomment line 25 and comment out line 22.

## Usage
*Make sure you are in a pipenv shell before running commands*
#### `spark-submit tst.py COMMAND [--ARGS] <INPUT>`

### Commands
sentiment-at-hour :
Input an hour of the day (0 - 23) and recieve the sentiment bias at this hour (sentiment in the range of -1.0 (most negative) to 1.0 (most positive)).
#### `spark-submit tst.py sentiment-at-hour <hour>`
---
sentiment-of-tweet : Input a string with a max of 140 characters, this will be parsed and the average sentiment of each word (if found in the data) will be returned (sentiment in the range of -1.0 (most negative) to 1.0 (most positive)).
#### `spark-submit tst.py sentiment-of-tweet "<tweet>"`
---
sentiment-of-word :
Input a word and if it exists in the data (and is not a [stop word](http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words)), return the average sentiment for this word (sentiment in the range of -1.0 (most negative) to 1.0 (most positive)).
#### `spark-submit tst.py sentiment-of-word <word>`
---
words-by-sentiment : Returns a list of words with the most positive/negative sentiments (sentiment in the range of -1.0 (most negative) to 1.0 (most positive)). The default is positive but you can get the most negative words with the `--negative` option. You can specify the minimum amount of occurences for a word to be returned with the `--min-samples` option. You can also specify the number of words to return with the `-max` option. 
#### `spark-submit tst.py words-by-sentiment [--positive|--negative] [--min-samples] <min>=2 [--max] <max>=20`
---
sentiment-of-user :
Input a twitter username and if it exists in the data, return the average sentiment of this user's tweets (sentiment in the range of -1.0 (most negative) to 1.0 (most positive)).
#### `spark-submit tst.py sentiment-of-user <username>`
---
users-by-sentiment : Returns a list of users with the most positive/negative sentiments (sentiment in the range of -1.0 (most negative) to 1.0 (most positive)). The default is positive but you can get the most negative users with the `--negative` option. You can specify the minimum amount of tweets a user must have to be returned, with the `--min-samples` option. You can also specify the number of users to return with the `-max` option. 
#### `spark-submit tst.py users-by-sentiment [--positive|--negative] [--min-samples] <min>=2 [--max] <max>=20`

### Data

The data used for this project is called Sentiment140 and was produced by Stanford University. You can [download it](http://help.sentiment140.com/for-students/) and unzip it into the `/data` folder.

Alternatively you can use any data you want as long as it has this csv format:

|**Column**|polarity of the tweet|id of the tweet|date of the tweet|query (lyx)|user that tweeted|text of the tweet|
|---       |---|---|---|---|---|---|
|**Format**|"(0 = negative, 2 = neutral, 4 = positive)"|"2087"|"EEE MMM dd HH:mm:ss zzz yyyy"|If there is no query, then this value is "NO_QUERY"|"robotickilldozr"|"Lyx is cool"|


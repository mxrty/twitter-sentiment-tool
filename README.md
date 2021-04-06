# twitter-sentiment-tool
A project built using Apache Spark's python library - PySpark\
Uses the [Sentiment140 dataset](http://help.sentiment140.com/for-students/)\
Can be operated using CLI built with Click

Developed and tested on Linux (Ubuntu 20.04)

## Installation
### Prerequisites:
These need to be installed on your machine (and create system PATH variables):
* Apache Spark 3.1.1 built for Hadoop 3.2+ [here](https://spark.apache.org/downloads.html)
* Java
* Python3

Then, [install pip](https://linuxize.com/post/how-to-install-pip-on-ubuntu-20.04/)

### Steps:

Install pipenv:

#### `pip3 install pipenv`

or 

#### `pip install pipenv`

Clone repository:

#### `git clone https://github.com/mxrty/twitter-sentiment-tool.git`

Install packages:

#### `pipenv install`

Run virtual-env shell:

#### `pipenv shell`

## Usage

#### `spark-submit tst.py COMMAND [--ARGS] <INPUT>`

### Commands
sentiment-at-hour :
Input an hour of the day (0 - 23) and recieve the sentiment bias at this hour.
#### `spark-submit tst.py sentiment-at-hour <hour>`
---
sentiment-of-word :
Input a word and if it exists in the data (and is not a [stop word](http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words)), return the average sentiment for this word
#### `spark-submit tst.py sentiment-of-word <word>`
---
sentiment-of-user :
Input a twitter username and if it exists in the data, return the average sentiment of this user's tweets
#### `spark-submit tst.py sentiment-of-user <username>`
---
words-by-sentiment : Returns a list of words with the most positive/negative sentiments. The default is positive but you can get the most negative words with the `--negative` option. You can specify the minimum amount of occurences for a word to be returned with the --`min-samples` option. You can also specify the number of words to return with the `-max` option. 
#### `spark-submit tst.py words-by-sentiment [--positive|--negative] [--min-samples] <min>=2 [--max] <max>=20`
---
users-by-sentiment : Returns a list of users with the most positive/negative sentiments. The default is positive but you can get the most negative users with the `--negative` option. You can specify the minimum amount of tweets a user must have to be returned, with the --`min-samples` option. You can also specify the number of users to return with the `-max` option. 
#### `spark-submit tst.py users-by-sentiment [--positive|--negative] [--min-samples] <min>=2 [--max] <max>=20`
---
sentiment-of-tweet : Input a string with a max of 140 characters, this will be parsed and the average sentiment of each word (if found in the data) will be returned.
#### `spark-submit tst.py sentiment-of-tweet "<tweet>"`

### Data

The data used for this project is called Sentiment140 and was produced by Stanford University. You can [download it](http://help.sentiment140.com/for-students/) and unzip it into the `/data` folder.

Alternatively you can use any data you want as long as it has this csv format:

|**Column**|polarity of the tweet|id of the tweet|date of the tweet|query (lyx)|user that tweeted|text of the tweet|
|---       |---|---|---|---|---|---|
|**Format**|"(0 = negative, 2 = neutral, 4 = positive)"|"2087"|"EEE MMM dd HH:mm:ss zzz yyyy"|If there is no query, then this value is "NO_QUERY"|"robotickilldozr"|"Lyx is cool"|


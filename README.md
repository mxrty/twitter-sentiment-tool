# twitter-sentiment-tool
A project built using Apache Spark's python library - PySpark\
Uses the [Sentiment140 dataset](http://help.sentiment140.com/for-students/)\
Can be operated using CLI built with Click

Developed and tested on Linux (Ubuntu 20.04) 
## Installation
### Prerequisites
These need to be installed on your machine (and create system PATH variables):
* Apache Spark 3.1.1 built for Hadoop 3.2+ [here](https://spark.apache.org/downloads.html)
* Java
* Python3

[Install pip](https://linuxize.com/post/how-to-install-pip-on-ubuntu-20.04/)

### `pip3 install pipenv`

or 

### `pip install pipenv`

Clone repository:

### `git clone https://github.com/mxrty/twitter-sentiment-tool.git`

### `pipenv install`

### `pipenv shell`

## Usage
### `spark-submit tst.py COMMAND [--ARGS] <INPUT>`

### Commands
### `spark-submit tst.py sentiment-at-hour <hour>` 

### `spark-submit tst.py sentiment-of-word <word>`

### `spark-submit tst.py sentiment-of-user <username>`

### `spark-submit tst.py words-by-sentiment [--positive|--negative] [--min-samples] <min>=2 [--max] <max>=20`

### `spark-submit tst.py users-by-sentiment [--positive|--negative] [--min-samples] <min>=2 [--max] <max>=20`

###`spark-submit tst.py sentiment-of-tweet "<tweet>"`
### Data
The data used for this project is called Sentiment140 and was produced by Stanford University. You can [download it](http://help.sentiment140.com/for-students/) and unzip it into the `/data` folder.

Alternatively you can use any data you want as long as it has this csv format:
|**Column**|polarity of the tweet|id of the tweet|date of the tweet|query (lyx)|user that tweeted|text of the tweet|
|---       |---|---|---|---|---|---|
|**Format**|"(0 = negative, 2 = neutral, 4 = positive)"|"2087"|"EEE MMM dd HH:mm:ss zzz yyyy"|If there is no query, then this value is "NO_QUERY"|"robotickilldozr"|"Lyx is cool"|


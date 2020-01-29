# Twitter-Climate-Change-Sentiment-Analysis

#### -- Project Status: Active

## Project Objective
The purpose of this project is to build a sentiment classifier for climate change related tweets. We are also interested in studying the correlation between climate change sentiment and daily global surface temperatures. 

### Methods Used
* Machine Learning 
* Data Visualization
* Data Cleaning
* Natural Language Processing

### Technologies 
* Python
* tweepy
* pandas
* scikit-learn
* Jupyter

## Project Description
Data cleaning: <br>
The labeled Kaggle dataset that we started with contained tweets with corrupted text, so we used Tweepy to access all the tweets by tweet id and retrieve the original clean text as well as additional features that could be of use. Around 30% of the tweets were no longer available, so we imputed the missing data and using natural language processing technqiues to clean the textual features. <br>

Model fitting: <br>
We trained Naive Bayes, Logisitic Regression, and Decision Tree models to the data for 3 class (1: supports the belief of man-made climate change, 0: neither supports nor refutes the belief of man-made climate change, -1: refutes the belief of man-made climate change) as well as a 2 class model omitting class 0 due to the ambiguity that we noticed in the content of many of the class 0 tweets. <br>

We are currently working on improving model performace through feature selection. Our next step will be to decide on a metric of extreme daily temperature and examine its correlation with sentiment towards climate change.


## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](https://github.com/lcoda/Twitter-Climate-Change-Sentiment-Analysis/tree/master/data) within this repo.
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)


## Featured Notebooks
* [Data Cleaning](link)
* [Data Visualization](link)
* [Model Fitting](link)


## Authors

**[Elizabeth Coda](https://github.com/lcoda): (elizabethcoda@berkeley.edu)** <br>
**[Juliette Franzman](https://github.com/juliettefranzman): (juliettef@berkeley.edu)**


## Acknowledgments

[README template](https://github.com/sfbrigade/data-science-wg/blob/master/dswg_project_resources/Project-README-template.md) thanks to the Data Science Working Group at Code for San Francisco. [Tweet sentiment data](https://www.kaggle.com/edqian/twitter-climate-change-sentiment-dataset) and [daily global surface temperature data](https://www.kaggle.com/noaa/noaa-global-surface-summary-of-the-day) were both downloaded from Kaggle.

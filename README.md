# Twitter-Climate-Change-Sentiment-Analysis

#### -- Project Status: Active

## Project Objective
The purpose of this project is to build a model that classifies the sentiment of tweets towards man-made climate change. We are also interested in studying the correlation between climate change sentiment and daily global surface temperature. 

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
The labeled Kaggle dataset that we started with contained tweets with corrupted text, so we used Tweepy to access all the tweets by tweet id and retrieve the original clean text as well as additional features that could be of use. Around 30% of the tweets were no longer available, so we imputed the missing data and used natural language processing techniques to clean the textual features. <br>

Model fitting: <br>
We trained Naive Bayes, Logisitic Regression, and Decision Tree models to predict three classes:
 - 1: supports the belief of man-made climate change
 - 0: neither supports nor refutes the belief of man-made climate change
 - -1: refutes the belief of man-made climate change
 
We also trained a two class model omitting class 0 due to the ambiguity that we noticed in the content of many of the class 0 tweets. <br>

We are currently working on improving model performance through feature selection. Our next step will be to decide on a metric of extreme daily temperature and examine its correlation with sentiment towards climate change.


## Featured Notebooks

* [Data Cleaning](https://github.com/lcoda/Twitter-Climate-Change-Sentiment-Analysis/blob/master/data_cleaning.ipynb)
* [Exploratory Data Analysis](https://github.com/lcoda/Twitter-Climate-Change-Sentiment-Analysis/blob/master/EDA.ipynb)
* [Model Fitting](https://github.com/lcoda/Twitter-Climate-Change-Sentiment-Analysis/blob/master/Initial%20Model%20Fitting.ipynb)


## Authors

**[Elizabeth Coda](https://github.com/lcoda): (elizabethcoda@berkeley.edu)** <br>
**[Juliette Franzman](https://github.com/juliettefranzman): (juliettef@berkeley.edu)**


## Acknowledgments

[README template](https://github.com/sfbrigade/data-science-wg/blob/master/dswg_project_resources/Project-README-template.md) thanks to the Data Science Working Group at Code for San Francisco. [Tweet sentiment data](https://www.kaggle.com/edqian/twitter-climate-change-sentiment-dataset) and [daily global surface temperature data](https://www.kaggle.com/noaa/noaa-global-surface-summary-of-the-day) were both downloaded from Kaggle.

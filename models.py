import numpy as np
from scipy.sparse import csr_matrix, hstack

import sklearn
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



def naive_bayes(text_array, class_vector, ngram=(1,1), maxwords=None,skvectorizer=CountVectorizer):
    """ 
    5 fold cross validation for Multinomial Naive Bayes classifier. 
    
    Parameters: 
    text_array (numpy array): cleaned text
    class_vector (numpy array): class labels
    ngram (tuple): n-gram range, default=(1,1)
    maxwords (int): maximum number of words features to use, default=None
    skvectorizer (sklearn.feature_extraction.text): vectorizer to use, default=CountVectorizer
  
    Returns: 
    float: Accuracy
    float: Precision
    float: Recall
    float: F score
  
    """
    #split data into 5 folds
    kf = KFold(n_splits=5, random_state = 0, shuffle=True)
    fold1, fold2, fold3, fold4, fold5 = kf.split(text_array)
    folds = [fold1, fold2, fold3, fold4, fold5]
    
    scores = []
    precision = []
    recall = []
    Fscore = []
    
    for f in folds:
        #split data into testing a training set 
        train, train_classes = text_array[f[0]], class_vector[f[0]]
        test, test_classes = text_array[f[1]], class_vector[f[1]]
        
        #vectorize the features
        vectorizer = skvectorizer(ngram_range=ngram, max_features=maxwords)
        train_v = vectorizer.fit_transform(train)
        test_v = vectorizer.transform(test)
        
        #return test_v,test_classes
        
        #run NB
        nb_classifier = MultinomialNB()
        nb_classifier.fit(train_v, train_classes)
        #print(nb_classifier.score(test_v, test_classes))
        #print(type(nb_classifier.score(test_v, test_classes)))
        
        # compute accuracy
        scores += [nb_classifier.score(test_v, test_classes)]
        
        # compute precision, recall, and fscore (macro-averaged)
        metrics = sklearn.metrics.precision_recall_fscore_support(test_classes,
                                                                  nb_classifier.predict(test_v),
                                                                  average='macro',
                                                                  zero_division=0)
        precision.append(metrics[0])
        recall.append(metrics[1])
        Fscore.append(metrics[2])
    
    return np.mean(scores), np.mean(precision), np.mean(recall), np.mean(Fscore)



def logistic(text_array, class_vector, numerical_matrix = None, ngram=(1,1), maxwords=None,skvectorizer=CountVectorizer):
    """ 
    5 fold cross validation for Logistic Regression classifier. 
    
    Parameters: 
    text_array (numpy array): cleaned text
    class_vector (numpy array): class labels
    numerical_matrix (numpy array): matrix of other numerical features, default=None
    ngram (tuple): n-gram range, default=(1,1)
    maxwords (int): maximum number of words features to use, default=None
    skvectorizer (sklearn.feature_extraction.text): vectorizer to use, default=CountVectorizer
  
    Returns: 
    float: Accuracy
    float: Precision
    float: Recall
    float: F score
  
    """
    #split data into 5 folds
    kf = KFold(n_splits=5, random_state = 0, shuffle=True)
    fold1, fold2, fold3, fold4, fold5 = kf.split(text_array)
    folds = [fold1, fold2, fold3, fold4, fold5]
    
    scores = []
    precision = []
    recall = []
    Fscore = []
    
    for f in folds:
        #split data into testing a training set 
        train_text, train_classes = text_array[f[0]], class_vector[f[0]] 
        test_text, test_classes = text_array[f[1]], class_vector[f[1]]
        
        #vectorize the features
        vectorizer = skvectorizer(ngram_range=ngram, max_features=maxwords)
        train = vectorizer.fit_transform(train_text)
        test = vectorizer.transform(test_text)
        
        #combine with rest of numerical data 
        if type(numerical_matrix) != type(None):
            train = hstack([train, csr_matrix(numerical_matrix[f[0]])])
            test = hstack([test, csr_matrix(numerical_matrix[f[1]])])
        
        #run logistic regression
        logistic_classifier = LogisticRegression(random_state=0, max_iter = 2000)
        logistic_classifier.fit(train, train_classes)
        
        # compute accuracy
        scores += [logistic_classifier.score(test, test_classes)]
        
        # compute precision, recall, and fscore (macro-averaged)
        metrics = sklearn.metrics.precision_recall_fscore_support(test_classes,
                                                                  logistic_classifier.predict(test),
                                                                  average='macro',
                                                                  zero_division=0)
        precision.append(metrics[0])
        recall.append(metrics[1])
        Fscore.append(metrics[2])
    
    return np.mean(scores), np.mean(precision), np.mean(recall), np.mean(Fscore)




def random_forest(text_array, class_vector, numerical_matrix = None, ngram=(1,1), maxwords=None, skvectorizer=CountVectorizer):
    """ 
    5 fold cross validation for Random Forest classifier. 
    
    Parameters: 
    text_array (numpy array): cleaned text
    class_vector (numpy array): class labels
    numerical_matrix (numpy array): matrix of other numerical features, default=None
    ngram (tuple): n-gram range, default=(1,1)
    maxwords (int): maximum number of words features to use, default=None
    skvectorizer (sklearn.feature_extraction.text): vectorizer to use, default=CountVectorizer
  
    Returns: 
    float: Accuracy
    float: Precision
    float: Recall
    float: F score
  
    """    
    #split data into 5 folds
    kf = KFold(n_splits=5, random_state=0, shuffle=True)
    fold1, fold2, fold3, fold4, fold5 = kf.split(text_array)
    folds = [fold1, fold2, fold3, fold4, fold5]
    
    scores = []
    precision = []
    recall = []
    Fscore = []
    
    for f in folds:
        #split data into testing a training set 
        train_text, train_classes = text_array[f[0]], class_vector[f[0]] 
        test_text, test_classes = text_array[f[1]], class_vector[f[1]]
        
        #vectorize the features
        vectorizer = skvectorizer(ngram_range=ngram, max_features=maxwords)
        train = vectorizer.fit_transform(train_text)
        test = vectorizer.transform(test_text)
        
        #combine with rest of numerical data 
        if type(numerical_matrix) != type(None):
            train = hstack([train, csr_matrix(numerical_matrix[f[0]])])
            test = hstack([test, csr_matrix(numerical_matrix[f[1]])])
        
        #run logistic regression
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(train, train_classes)
        
        # compute accuracy
        scores += [rf_classifier.score(test, test_classes)]
        
        # compute precision, recall, and fscore (macro-averaged)
        metrics = sklearn.metrics.precision_recall_fscore_support(test_classes,
                                                                  rf_classifier.predict(test),
                                                                  average='macro',
                                                                  zero_division=0)
        precision.append(metrics[0])
        recall.append(metrics[1])
        Fscore.append(metrics[2])
    
    return np.mean(scores), np.mean(precision), np.mean(recall), np.mean(Fscore)
    
    
    
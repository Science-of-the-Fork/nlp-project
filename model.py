# Credential imports
from env import github_token, github_username

# Scraping related imports
import requests
from bs4 import BeautifulSoup

# NLP related imports
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import unicodedata
import re

# General data manipulation imports
import pandas as pd
import numpy as np
from time import strftime
from typing import Dict, List, Optional, Union, cast
import os
import json

# For splitting the data
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import scipy.stats as stats
from scipy.stats import  chi2_contingency
import matplotlib.pyplot as plt


import seaborn as sns

# Quiet all warnings
import warnings
warnings.filterwarnings('ignore')


############### PREPROCESS FUNCTION ##############

def preprocess(train, validate, test):
    """
    Takes in train, validate, and test dataframes then appends columns the counts of how often they show up in 
    each readme to each dataframe, uses these columns to create 'X_' dataframes and uses the 'language' column to
    create 'y_' dataframes. outputs X_ and y_ dataframes for train, validate, and test.
    """
    # List of significant words to make columns for
    significant_words = ['learning', 'test', 'library', 'create', 'line']

    # Iterate through the list of significant words and for each word...
    for word in significant_words:
        # Create a column in the dataframe that holds the count of how many times that word was used in each readme
        train[word] = train["lemmatized"].apply(lambda x: x.count(word))
    # assign just the columns of significant word counts to a dataframe called 'X_train'
    X_train = train[significant_words]
    # Assign the 'language' column to a dataframe called 'y_train'
    y_train = train["language"]
    
    # Iterate through the list of significant words and for each word...
    for word in significant_words:
        # Create a column in the dataframe that holds the count of how many times that word was used in each readme
        validate[word] = validate["lemmatized"].apply(lambda x: x.count(word))
    # assign just the columns of significant word counts to a dataframe called 'X_validate'
    X_validate = validate[significant_words]
    # Assign the 'language' column to a dataframe called 'y_validate'
    y_validate = validate["language"]
    
    # Iterate through the list of significant words and for each word...
    for word in significant_words:
        # Create a column in the dataframe that holds the count of how many times that word was used in each readme
        test[word] = test["lemmatized"].apply(lambda x: x.count(word))
    # assign just the columns of significant word counts to a dataframe called 'X_test'
    X_test = test[significant_words]
    # Assign the 'language' column to a dataframe called 'y_test'
    y_test = test["language"]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test


############## BASELINE MODEL FUNCTION ###########

def baseline(train):  
    """
    Creates the baseline model and displays the accuracy
    """
    most_common_class = train["language"].value_counts().idxmax()
    most_common_frequency = train["language"].value_counts().max()

    baseline_accuracy = most_common_frequency / len(train)

    print(f"Most Common Class: {most_common_class}")
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}")
    

############# DECISION TREE FUNCTION ##############
  
def decision_tree(X_train, y_train, X_validate, y_validate): 
    """
    Takes in dataframe, builds a decision tree classifier, and evaluates it on the train and validate data.
    """
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_validate)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_validate, y_val_pred)

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}")

    
################## RANDOM FOREST FUNCTION ###############

def random_forest(X_train, y_train, X_validate, y_validate): 
    """
    Takes in dataframe, builds a random forest classifier, and evaluates it on the train and validate data.
    """
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_validate)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_validate, y_val_pred)

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}")


#################### KNN FUNCTION #####################
    
def knn(X_train, y_train, X_validate, y_validate):  
    """
    Takes in dataframe, builds a knn classifier, and evaluates it on the train and validate data.
    """
    k = 3  
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_validate)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_valdate, y_val_pred)

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}")


############## LOGISTIC REGRESSION ####################
 
def logistic_regresssion(X_train, y_train, X_validate, y_validate): 
    """
    Takes in dataframe, builds a logistic regression classifier, and evaluates it on the train and validate data.
    """
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    print(f"Train Accuracy: {train_accuracy:.2f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}")

    
################# TEST RANDOM FOREST FUNCTION ##################

def test_random_forest(X_train, y_train, X_test, y_test): 
    """
    Takes in dataframe, builds a knn classifier, and evaluates it on the test data.
    """
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_test_pred = clf.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)


    print(f"Test Accuracy: {test_accuracy:.2f}")
    
    
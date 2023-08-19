############## INTIAL IMPORTS ###############

import re
import unicodedata
import pandas as pd
import nltk
from wordcloud import WordCloud

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import scipy.stats as stats
from scipy.stats import  chi2_contingency
from scipy.stats import ttest_ind

from sklearn.feature_extraction.text import CountVectorizer

############ DATA PREVIEW FUNCTION ############

def data_preview(train):
    """
    Displays the head of the train column, with the removal of the second index column
    """
    display(train.drop(columns='Unnamed: 0').head())


############ GET DATA FUNCTION ############

def get_data(train):
    """
    Stores the language and lemmatized columns in the 'df' dataframe
    """
    df = train[["language", "lemmatized"]]
    return df


######## DISPLAY PREVALENCE FUNCTION #########

def display_prevalence(df):
    """
    Displays the prevalence of python versus javascript in a barplot
    """
    print(df.language.value_counts())
    print(df.language.value_counts(normalize=True))
    plt.figure(figsize=(6,3))
    df.language.value_counts().plot(kind="bar");
    
    
########### MOST FREQUENT PYTHON FUNCTION #########

def most_frequent_python(df):
    """
    Displays the most frequently occuring words in python readmes
    """
    py = df.lemmatized[df.language == "Python"]
    plt.figure(figsize=(8,3))
    pd.Series(" ".join(py).split()).value_counts().head(20).plot(kind="bar")
    plt.title("Python most frequently occuring words")
    plt.ylabel("frequency");

    
########### MOST FREQUENT PYTHON FUNCTION #########

def most_frequent_javascript(df):
    """
    Displays the most frequently occuring words in javascript readmes
    """
    js = df.lemmatized[df.language == "JavaScript"]
    plt.figure(figsize=(8,3))
    pd.Series(" ".join(js).split()).value_counts().head(20).plot(kind="bar")
    plt.title("JavaScript most frequently occuring words")
    plt.ylabel("frequency");

    
########### MOST FREQUENT OVERALL FUNCTION #########

def most_frequent_overall(df):
    """
    Displays the most frequently occuring words in readmes overall
    """
    py_js = df.lemmatized
    plt.figure(figsize=(8,3))
    pd.Series(" ".join(py_js).split()).value_counts().head(20).plot(kind="bar")
    plt.title("Most frequently occuring words overall")
    plt.ylabel("frequency");

    
########### MAKE FREQUENCY DF FUNCTION ###############

def make_frequency_df(df):
    """
    Takes in the dataframe of training data and constructs a frequency table of python and javascript words
    """
    py = df.lemmatized[df.language == "Python"]
    js = df.lemmatized[df.language == "JavaScript"]
    py_js = df.lemmatized
    
    py_df = pd.Series(" ".join(py).split()).value_counts()
    js_df = pd.Series(" ".join(js).split()).value_counts()
    all_df = pd.Series(" ".join(py_js).split()).value_counts()

    freq_df = pd.concat([py_df, js_df, all_df], axis=1).set_axis(["py", "js", "all"], axis=1)
    freq_df = freq_df.fillna(0)
    
    return freq_df
    
    
############## PYTHON FREQUENT EXLUSIVE WORDS FUNCTION #######
    
def python_frequent_exclusive_words(freq_df):
    """
    Plots the most frequently occuring words that are exclusive to python
    """
    plt.figure(figsize=(8,3))

    # words exclusive to python
    freq_df["py"][freq_df["js"] == 0].head(20).plot(kind="bar")
    plt.title("Python exclusive occuring words")
    plt.ylabel("frequency");
    
    
############ JAVASCRIPT FREQUENT EXLUSIVE WORDS FUNCTION #######
    
def javascript_frequent_exclusive_words(freq_df):
    """
    Plots the most frequently occuring words that are exclusive to python
    """
    plt.figure(figsize=(8,3))
    # words exclusive to python
    freq_df["js"][freq_df["py"] == 0].head(20).plot(kind="bar")
    plt.title("JavaScript exclusive occuring words")
    plt.ylabel("frequency");
    

#################### STATS TEST TOP 80 ######################
    
def stats_test_top_80(train):
    """
    Takes in the training dataframe. Conducts an independent t-test on the dataframe, and displays the words that are significantly correlated to language along with their test statistcs
    """
    df = train

    vectorizer = CountVectorizer()

    X_train = vectorizer.fit_transform(df["lemmatized"])

    word_frequencies_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())

    word_frequencies_total = word_frequencies_df.sum()

    top_n = 80
    top_words = word_frequencies_total.sort_values(ascending=False).index[:top_n]

    for word in top_words:
        word_frequencies_by_language = {language: [] for language in df["language"].unique()}
        for index, row in df.iterrows():
            words = row["lemmatized"].split()
            if word in words:
                word_frequencies_by_language[row["language"]].append(words.count(word))

        language_groups = [word_frequencies_by_language[language] for language in df["language"].unique()]
        statistic, p_value = ttest_ind(*language_groups)


        alpha = 0.05
        if p_value < alpha:
            print(f"Word: '{word}'")
            print("T-Test Statistic:", statistic)
            print("P-value:", p_value)
            print(f"There is a significant difference in word '{word}' frequencies among languages.\n")
        
        

#################################################################
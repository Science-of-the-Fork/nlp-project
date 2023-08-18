from bs4 import BeautifulSoup
import requests
import pandas as pd
import acquire as a
import re
import unicodedata
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os
import warnings
warnings.filterwarnings('ignore')



def basic_clean(text):
    
    text = text.lower()
    
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    text = re.sub(r'[^a-zA-Z0-9\s\']', '', text)
    
    return text

def tokenize(input_string):

    words = input_string.split()
    return words


def stem(text):
    
    stemmer = PorterStemmer()
    
    words = word_tokenize(text)

    stemmed_words = [stemmer.stem(word) for word in words]

    stemmed_text = ' '.join(stemmed_words)
    
    return stemmed_text

def lemmatize(text):
  
    lemmatizer = WordNetLemmatizer()
  
    words = word_tokenize(text)
  
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    
    return lemmatized_text


def remove_stopwords(text, extra_words=None, exclude_words=None):
   
    stop_words = set(stopwords.words("english"))

    if extra_words:
        stop_words.update(extra_words)

    if exclude_words:
        stop_words.difference_update(exclude_words)

    words = text.split()
  
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    filtered_text = ' '.join(filtered_words)
    
    return filtered_text

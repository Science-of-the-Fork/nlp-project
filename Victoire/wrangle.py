######################## INITIAL IMPORTS #######################

import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd
import acquire
from time import strftime

import warnings
warnings.filterwarnings('ignore')


###################### SPLIT READMES FUNCTION ###########################

def split_readmes(df):
    """
    Takes in a dataframe and performs a 70/15/15 split. Outputs a train, validate, and test dataframe
    """
    # Perfrom a 70/15/15 split
    train_val, test = train_test_split(df, test_size=.15, random_state=123)
    train, validate = train_test_split(train_val, test_size=.17, random_state=123)
    
    return train, validate, test


###################### BASIC CLEAN FUNCTION ######################

def clean(string):
    """
    This function puts a string in lowercase, normalizes any unicode characters, removes anything that         
    isn't an alphanumeric symbol or single quote.
    """
    # Normalize unicode characters
    string = unicodedata.normalize('NFKD', string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    
    # Remove unwanted characters and put string in lowercase
    string = re.sub(r"[^\w0-9'\s]", '', string).lower()
            
    return string


##################### TOKENIZE FUNCTION #########################

def tokenize(string):
    """
    Takes in a string and tokenizes it. Returns the tokenized string.
    """
    # Build the tokenizer
    tokenizer = nltk.tokenize.ToktokTokenizer()
    
    # Tokenize the string with the tok-tok-tokenizerrr
    string = tokenizer.tokenize(string, return_str = True)
    
    return string


###################### STEM FUNCTION ############################

def stem(string):
    """
    This function takes in a string, stems it, and returns a stemmed version of the original string
    """
    # Build the stemmer
    stemmer = nltk.porter.PorterStemmer()
    
    # Use the stemmer on each word in the string and append to the results list
    results = []
    for word in string.split():
        results.append(stemmer.stem(word))
        
    # Convert back into a string
    string = ' '.join(results)
    
    return string


##################### LEMMATIZE FUNCTION ########################

def lemmatize(string):
    """
    This function takes in a string, lemmatizes each word, and returns a lemmatized version of the orignal string
    """
    # Build the lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    results = []
    for word in string.split():
        results.append(lemmatizer.lemmatize(word))
    
    # Convert back into a string
    string = ' '.join(results)

    return string


######################## REMOVE STOPWORDS FUNCTION ###########################

def remove_stopwords(string, extra_words=None, exclude_words=None):
    """
    Takes in a string, with optional arguments for words to add to stock stopwords and words to ignore in the 
    stock list removes the stopwords, and returns a stopword free version of the original string
    """
    # Get the list of stopwords from nltk
    stopword_list = stopwords.words('english')
    
    # Create a set of stopwords to exclude
    excluded_stopwords = set(exclude_words) if exclude_words else set()
    
    # Include any extra words in the stopwords to exclude
    stopwords_to_exclude = set(stopword_list) - excluded_stopwords
    
    # Add extra words to the stopwords set
    stopwords_to_exclude |= set(extra_words) if extra_words else set()
    
    # Tokenize the input string
    words = string.split()
    
    # Filter out stopwords from the tokenized words
    filtered_words = [word for word in words if word not in stopwords_to_exclude]
    
    # Convert back to string
    string = ' '.join(filtered_words)
    
    return string


    
# import acquire as a
# from env import github_token, github_username
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split
import os


# import sys
# sys.path.append("..")
# import acquire as a
# from env import github_token, github_username
"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import json
from typing import Dict, List, Optional, Union, cast


"""
A module for obtaining repo readme and language data from the github API.
Before using this module, read through it, and follow the instructions marked
TODO.
After doing so, run it like this:
    python acquire.py
To create the `data.json` file that contains the data.
"""
import json
from typing import Dict, List, Optional, Union, cast
from env import github_token, github_username

# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.

REPOS = []
 

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )
    
    
############### ACQUIRE HELPER FUNCTIONS ###################

def github_api_request(url: str) -> Union[List, Dict]:
    response = requests.get(url, headers=headers)
    response_data = response.json()
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
    return response_data


def get_repo_language(repo: str) -> str:
    url = f"https://api.github.com/repos/{repo}"
    repo_info = github_api_request(url)
    if type(repo_info) is dict:
        repo_info = cast(Dict, repo_info)
        return repo_info.get("language", None)
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    url = f"https://api.github.com/repos/{repo}/contents/"
    contents = github_api_request(url)
    if type(contents) is list:
        contents = cast(List, contents)
        return contents
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )


def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and
    returns the url that can be used to download the repo's README file.
    """
    for file in files:
        if file["name"].lower().startswith("readme"):
            return file["download_url"]
    return ""


def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a
    dictionary with the language of the repo and the readme contents.
    """
    contents = get_repo_contents(repo)
    readme_contents = requests.get(get_readme_download_url(contents)).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loop through all of the repos and process them. Returns the processed data.
    """
    return [process_repo(repo) for repo in REPOS]


#################### GET GITHUB PYTHON FUNCTION ######################

def get_github_python_data():
    # repo = 'python/cpython' # repository identification
    # authentications
    headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

    REPO_NAME = []
    for i in range(1, 10):
        # url to python repos
        python_url= f'https://github.com/search?o=desc&q=stars%3A%3E1+language%3APython&s=forks&type=Repositories&p=1&l=python={i}'

        # get json rescponce
        res = github_api_request(python_url)
        temp_name = [REPO_NAME.append(x['hl_name']) for x in res['payload']['results']]

    repo_names = []
    url_link = []
    readme_con = []
    repo_language = []
    rst_file= []
    for page_repo in REPO_NAME:
        repo_content = get_repo_contents(page_repo)
        # locate the Readme.rst file link
        for ele in range(len(repo_content)):
            link = repo_content[ele]["html_url"]
            match = re.search(f"README", link)
            if match:
                rst_file = link
                break
        # get the readme request
        readme_res = requests.get(rst_file)

        if readme_res.status_code == 200:
            # find read me content
            soup = BeautifulSoup(readme_res.text, 'html.parser')
            readme_content = soup.get_text()

            # extract noisy charactures from the content
            pattern = r'"richText":"(.*?)"\s*,\s*"renderedFileInfo"'
            matches = re.findall(pattern, readme_content)
            if len(matches) == 1:
                extracted_content = matches[0].replace("\\n","")

                # get repo language
                repo_lang = get_repo_language(page_repo)

                # url and readme content of all repo
                repo_names.append(page_repo)
                url_link.append(rst_file)
                readme_con.append(extracted_content)
                repo_language.append(repo_lang)
        else:
            print("Failed to connect...!")
        
    results = pd.DataFrame(repo_names, columns=["repo_name"]).assign(url = url_link, 
                                                            language = repo_language,
                                                            readme_content = readme_con)
    results.to_csv("python_data.csv", mode= "w")
    return results

#################### GET GITHUB JAVA SCRIPT FUNCTION ######################

def get_github_java_script_data():
    # repo = 'python/cpython' # repository identification
    # authentications
    headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

    REPO_NAME = []
    for i in range(1, 10):
        # url to python repos
        python_url= f'https://github.com/search?o=desc&q=stars%3A%3E1+language%3AJavaScript&s=forks&type=Repositories&l=JavaScript&p={i}'

        # get json rescponce
        res = github_api_request(python_url)
        # repo_content = get_repo_contents(repo)
        temp_name = [REPO_NAME.append(x['hl_name']) for x in res['payload']['results']]

    repo_names = []
    url_link = []
    readme_con = []
    repo_language = []
    rst_file= []
    for page_repo in REPO_NAME:
        repo_content = get_repo_contents(page_repo)
        # locate the Readme.rst file link
        for ele in range(len(repo_content)):
            link = repo_content[ele]["html_url"]
            match = re.search(f"README", link)
            if match:
                rst_file = link
                break
        # get the readme request
        readme_res = requests.get(rst_file)

        if readme_res.status_code == 200:
            # find read me content
            soup = BeautifulSoup(readme_res.text, 'html.parser')
            readme_content = soup.get_text()

            # extract noisy charactures from the content
            pattern = r'"richText":"(.*?)"\s*,\s*"renderedFileInfo"'
            matches = re.findall(pattern, readme_content)
            if len(matches) == 1:
                extracted_content = matches[0].replace("\\n","")

                # get repo language
                repo_lang = get_repo_language(page_repo)

                # url and readme content of all repo
                repo_names.append(page_repo)
                url_link.append(rst_file)
                readme_con.append(extracted_content)
                repo_language.append(repo_lang)
        else:
            print("Failed to connect...!")
            
    results = pd.DataFrame(repo_names, columns=["repo_name"]).assign(url = url_link, 
                                                            language = repo_language,
                                                            readme_content = readme_con)
    results.to_csv("java_script_data.csv", mode= "w")
    return results

if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)
    
    
######################## ACQUIRE READMES FUNCTION #####################

def acquire_readmes():
    """
    return existing data from csv in current working directory if the file exists,
    else with web scraping methods to get the data
        data_: 1 for python data, and 2 for java-script data
    """
    py_file = "python_data.csv"
    js_file = "java_script_data.csv"
    
    if os.path.exists(py_file):
        if os.path.exists(js_file):
            python_df = pd.read_csv(py_file)
            js_df = pd.read_csv(js_file)
            df = pd.concat([python_df, js_df], axis=0)
            print("returning python and Java-script data")
            return df
        else: 
            return pd.concat([python_df, get_github_java_script_data()], axis=0)
    
    
########################## PREPARE FUNCTION #################################
def clean(string):
    """
    This function puts a string in lowercase, normalizes any unicode characters, removes anything that         
    isn't an alphanumeric symbol or single quote.
    """
    # Normalize unicode characters
    string = unicodedata.normalize('NFKD', string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    
    # Remove unwanted characters and put string in lowercase
    string = re.sub(r"[^\w0-9'\s]", '', string).lower()
            
    return string

def prep_readmes(df, col="readme_content"):
    """
    Takes in the dataframe and the column name that contains the corpus data, creates a column of cleaned data, then uses that 
    to create a column without stopwords that is lemmatized, performs a train-validate-test split, performs an x-y split, and
    returns x and y train, x and y validate, and x and y test.
    """
    # Create the cleaned column

    cleaned_row = []
    for i in df.readme_content.values:
        cleaned_row.append(clean(i))
    df = df.assign(cleaned_content=cleaned_row)
#     df['cleaned'] = df[col].apply(lambda x: clean(x))
    df['lemmatized'] = df['cleaned_content'].apply(lambda x: lemmatize(remove_stopwords(x)))
    
    # Split the dataframe (70/15/15)
    train, validate, test = split_readmes(df)
    
#     # perform x-y split
#     x_train, y_train = train.drop(columns=('language')), train.language
#     x_validate, y_validate = validate.drop(columns=('language')), validate.language
#     x_test, y_test = test.drop(columns=('language')), test.language
    
    return train, validate, test
    
    
######################### WRANGLE READMES ##################################
                               
def wrangle_readmes():
    train, validate, test = prep_readmes(acquire_readmes(), 'readme_content')
    return train, validate, test


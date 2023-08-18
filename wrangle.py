######################## INITIAL IMPORTS #######################

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

# Quiet all warnings
import warnings
warnings.filterwarnings('ignore')



############### ACQUIRE HELPER FUNCTIONS ###################

################# INITIALIZE ACQUIRE CREDENTIALS ##################    

REPOS = []
 

headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )
    
    
############## GITHUB API REQUEST FUNCTION #################

def github_api_request(url: str) -> Union[List, Dict]:
    """
    Takes in a url and sends a request to github using the credentials stored within the headers variable, converts the data
    that's returned into .json format, raises an error if the response code comes back as other than 200 and displays the error,
    returns the content of the webpage in json format. Used as a helper function in the 'get_repo_language()' and 
    'get_repo_contents()' functions.
    """
    # Request the webpage data
    response = requests.get(url, headers=headers)
    
    # Convert to .json format
    response_data = response.json()
    
    # If the status comes back other than successful, raise an error and display the response code
    if response.status_code != 200:
        raise Exception(
            f"Error response from github api! status code: {response.status_code}, "
            f"response: {json.dumps(response_data)}"
        )
        
    # Return the response code
    return response_data


################# GET REPO LANGUAGE FUNCTION ##############

def get_repo_language(repo: str) -> str:
    """
    Takes in the repo name as a string. Sends an api request to the github server using the 'github_api_request()' function. 
    Returns the repo language.
    """
    # Construct the URL by incorporating the repo name to the format string
    url = f"https://api.github.com/repos/{repo}"
    
    # Send an API request to the github server using 'github_api_request()' as a helper function
    repo_info = github_api_request(url)
    
    # Check if the API response is a dictionary
    if type(repo_info) is dict:
       
        # Cast the response to a dictionary
        repo_info = cast(Dict, repo_info)
        
        # Return the value of the language key or 'None' if it isn't found
        return repo_info.get("language", None)
    
    # Raise an exception if the response comes back in the wrong format
    raise Exception(
        f"Expecting a dictionary response from {url}, instead got {json.dumps(repo_info)}"
    )


#################### GET REPO CONTENTS FUNCTION #############

def get_repo_contents(repo: str) -> List[Dict[str, str]]:
    """
    Takes in the repo name as a string. Sends an api request to the github server using the 'github_api_request()' function. 
    Returns the repo contents as a list.
    """
    # Construct the URL by incorporating the repo name to the format string
    url = f"https://api.github.com/repos/{repo}/contents/"
    
    # Send an API request to the github server using 'github_api_request()' as a helper function
    contents = github_api_request(url)
    
    # Check if the API response is a list
    if type(contents) is list:
        
        # Cast contents to a list
        contents = cast(List, contents)
        return contents
    
    # Raise an exception if the response comes back in the wrong format
    raise Exception(
        f"Expecting a list response from {url}, instead got {json.dumps(contents)}"
    )

    
################ GET README DOWNLOAD URL FUNCTION ##############

def get_readme_download_url(files: List[Dict[str, str]]) -> str:
    """
    Takes in a response from the github api that lists the files in a repo and returns the url that can be used to 
    download the repo's README file.
    """
    # Iterate through the dictionary files in the list
    for file in files:
        
        # If the contents of the 'name' key start with 'readme'...
        if file["name"].lower().startswith("readme"):
            
            # Return the contents of the 'download_url' key
            return file["download_url"]
    
    # Return an empty string
    return ""


###################### PROCESS REPO FUNCTION ###################

def process_repo(repo: str) -> Dict[str, str]:
    """
    Takes a repo name like "gocodeup/codeup-setup-script" and returns a dictionary with the language of the repo and 
    the readme contents.
    """
    # Use the get_repo_contents helper function to return the repo contents as a list of dictionaries
    contents = get_repo_contents(repo)
    
    # Use the 'get_readme_download_url()' helper function to request the readme site and store the text in the 
    # 'readme_contents' variable
    readme_contents = requests.get(get_readme_download_url(contents)).text
    
    # Return a dictionary of the repo name, repo language, and readme contents
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


################## SCRAPE GITHUB DATA FUNCTION ######################

def scrape_github_data() -> List[Dict[str, str]]:
    """
    Loops through all of the repos and process them. Returns the processed data in a list of dictionaries
    """
    # Use list comprehension to return a list of dictionaries containing the desired info for all readmes
    return [process_repo(repo) for repo in REPOS]


#################### GET GITHUB PYTHON FUNCTION ######################

def get_github_python_data():
    """
    Requests the first 10 pages from the forked repositories list on github filtered to repos written in python, collects the
    urls, repo titles, repo contents, repo language labels, and links to the readme files. Info is saved to a .csv and a 
    dataframe of the collected data is returned.
    """
    # Load github credentials into a dictionary called 'headers'
    headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}
    
    # Initialize the repo name list to hold repo names
    REPO_NAME = []
    
    # Iterate through 9 pages of the github forked repositories list filtered to repos that use python
    for i in range(1, 10):
        
        # Construct the url to python repos using a template string updated by the i variable in each iteration of the loop
        python_url= f'https://github.com/search?o=desc&q=stars%3A%3E1+language%3APython&s=forks&type=Repositories&p=1&l=python={i}'

        # Get the json response using the 'github_api_request()' helper function
        res = github_api_request(python_url)
        
        # Using list comprehension, collect all repo names from the page of results under consideration
        temp_name = [REPO_NAME.append(x['hl_name']) for x in res['payload']['results']]
    
    # Initialize lists to hold results of the next for-loop
    repo_names = []
    url_link = []
    readme_con = []
    repo_language = []
    rst_file= []
    
    # Iterate through the list of repo names collected in the last for-loop
    for page_repo in REPO_NAME:
        
        # Collect the contents of the page under consideration and store in the 'repo_content' variable
        repo_content = get_repo_contents(page_repo)
        
        # Iterate for the length of the list of repo content dictionaries 
        for ele in range(len(repo_content)):
            
            # Store the contents of the 'html_url' key for the dictionary under consideration in the 'link' variable
            link = repo_content[ele]["html_url"]
            
            # Search the link for the term 'README' and store a boolean of the results in the 'match' variable
            match = re.search(f"README", link)
            
            # If a match was found, store it in the 'rst_file' variable
            if match:
                rst_file = link
                break
                
        # Request the README URL that was just found and stored in 'rst_file'
        readme_res = requests.get(rst_file)

        # Check if the HTML response was successful
        if readme_res.status_code == 200:
            
            # Store a beautiful soup format of the HTML response in the 'soup' variable
            soup = BeautifulSoup(readme_res.text, 'html.parser')
            
            # Get the readme text out of the HTML response and store it in the 'readme_content' variable
            readme_content = soup.get_text()

            # Use regex to extract noisy charactures from the content, store in the 'pattern' variable
            pattern = r'"richText":"(.*?)"\s*,\s*"renderedFileInfo"'
            
            # Store a list of all pattern matches in the 'matches' variable
            matches = re.findall(pattern, readme_content)
            
            # If there is exactly one match...
            if len(matches) == 1:
                
                # Remove all line breaks from the match and store in the 'extracted_content' variable
                extracted_content = matches[0].replace("\\n","")

                # Get repo language and store in the 'repo_lang' variable
                repo_lang = get_repo_language(page_repo)

                # Add the page_repo, rst_file, extracted content, and repo language, to the lists initialized before the 
                # current for-loop
                repo_names.append(page_repo)
                url_link.append(rst_file)
                readme_con.append(extracted_content)
                repo_language.append(repo_lang)
                
        # If the connection was unsuccessful, print 'Failed to connect...!'
        else:
            print("Failed to connect...!")
        
    # Create a dataframe using repo names, and assign the associated information for each repo
    results = pd.DataFrame(repo_names, columns=["repo_name"]).assign(url = url_link, 
                                                            language = repo_language,
                                                            readme_content = readme_con)
    # Save the results to 'python_data.csv'
    results.to_csv("python_data.csv", mode= "w")
    
    # Return the dataframe of collected repo data
    return results


#################### GET GITHUB JAVA SCRIPT FUNCTION ######################

def get_github_java_script_data():
    """
    Requests the first 10 pages from the forked repositories list on github filtered to repos written in javascript, collects the
    urls, repo titles, repo contents, repo language labels, and links to the readme files. Info is saved to a .csv and a 
    dataframe of the collected data is returned.
    """
    # Load github credentials into a dictionary called 'headers'
    headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}
    
    # Initialize the repo name list to hold repo names
    REPO_NAME = []
    
    # Iterate through 9 pages of the github forked repositories list filtered to repos that use javascript
    for i in range(1, 10):
        
        # Construct the url to javascript repos using a template string updated by the i variable in each iteration of the loop
        python_url= f'https://github.com/search?o=desc&q=stars%3A%3E1+language%3AJavaScript&s=forks&type=Repositories&l=JavaScript&p={i}'

         # Get the json response using the 'github_api_request()' helper function
        res = github_api_request(python_url)
        
        # Using list comprehension, collect all repo names from the page of results under consideration
        temp_name = [REPO_NAME.append(x['hl_name']) for x in res['payload']['results']]
    
    # Initialize lists to hold results of the next for-loop
    repo_names = []
    url_link = []
    readme_con = []
    repo_language = []
    rst_file= []
    
    # Iterate through the list of repo names collected in the last for-loop
    for page_repo in REPO_NAME:
        
        # Collect the contents of the page under consideration and store in the 'repo_content' variable
        repo_content = get_repo_contents(page_repo)
        
        # Iterate for the length of the list of repo content dictionaries 
        for ele in range(len(repo_content)):
            
            # Store the contents of the 'html_url' key for the dictionary under consideration in the 'link' variable
            link = repo_content[ele]["html_url"]
            
            # Search the link for the term 'README' and store a boolean of the results in the 'match' variable
            match = re.search(f"README", link)
            
            # If a match was found, store it in the 'rst_file' variable
            if match:
                rst_file = link
                break
        
        # Request the README URL that was just found and stored in 'rst_file'
        readme_res = requests.get(rst_file)

        # Check if the HTML response was successful
        if readme_res.status_code == 200:
            
            # Store a beautiful soup format of the HTML response in the 'soup' variable
            soup = BeautifulSoup(readme_res.text, 'html.parser')
            
            # Get the readme text out of the HTML response and store it in the 'readme_content' variable
            readme_content = soup.get_text()

            # Use regex to extract noisy charactures from the content, store in the 'pattern' variable
            pattern = r'"richText":"(.*?)"\s*,\s*"renderedFileInfo"'
            
            # Store a list of all pattern matches in the 'matches' variable
            matches = re.findall(pattern, readme_content)
            
            # If there is exactly one match...
            if len(matches) == 1:
                
                # Remove all line breaks from the match and store in the 'extracted_content' variable
                extracted_content = matches[0].replace("\\n","")

                # Get repo language and store in the 'repo_lang' variable
                repo_lang = get_repo_language(page_repo)

                # Add the page_repo, rst_file, extracted content, and repo language, to the lists initialized before the 
                # current for-loop
                repo_names.append(page_repo)
                url_link.append(rst_file)
                readme_con.append(extracted_content)
                repo_language.append(repo_lang)
        
        # If the connection was unsuccessful, print 'Failed to connect...!'
        else:
            print("Failed to connect...!")
            
    # Create a dataframe using repo names, and assign the associated information for each repo
    results = pd.DataFrame(repo_names, columns=["repo_name"]).assign(url = url_link, 
                                                            language = repo_language,
                                                            readme_content = readme_con)
    
    # Save the results to 'java_script_data.csv'
    results.to_csv("java_script_data.csv", mode= "w")
    
    # Return the dataframe of collected repo data
    return results
    
    
######################## ACQUIRE READMES FUNCTION #####################

def acquire_readmes():
    """
    Return existing data from CSV in the current working directory if the file exists,
    else use web scraping methods to get the data.
    data_: 1 for python data, and 2 for java-script data
    """
    py_file = "python_data.csv"
    js_file = "java_script_data.csv"

    if os.path.exists(py_file):
        python_df = pd.read_csv(py_file)

        if os.path.exists(js_file):
            js_df = pd.read_csv(js_file)
            df = pd.concat([python_df, js_df], axis=0)
            print("Returning Python and Java-script data")
            return df
        else:
            print("Getting Java-script data via scraping")
            js_df = get_github_java_script_data()
            df = pd.concat([python_df, js_df], axis=0)
            return df
    else:
        print("Getting both Python and Java-script data via scraping")
        python_df = get_github_python_data()
        js_df = get_github_java_script_data()
        df = pd.concat([python_df, js_df], axis=0)
        return df


##################### PREPARE HELPER FUNCTIONS #########################

###################### SPLIT READMES FUNCTION ###########################

def split_readmes(df):
    """
    Takes in a dataframe and performs a 70/15/15 split. Outputs a train, validate, and test dataframe
    """
    # Perfrom a 70/15/15 split
    train_val, test = train_test_split(df, test_size=.15, random_state=123)
    train, validate = train_test_split(train_val, test_size=.17, random_state=123)
    
    # Return the dataframe slices
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
    # Build the stemmer object
    stemmer = nltk.porter.PorterStemmer()
    
    # Use the stemmer on each word in the string and append to the results list
    results = []
    for word in string.split():
        results.append(stemmer.stem(word))
        
    # Convert back into a string
    string = ' '.join(results)
    
    # Return the result string
    return string


##################### LEMMATIZE FUNCTION ########################

def lemmatize(string):
    """
    This function takes in a string, lemmatizes each word, and returns a lemmatized version of the orignal string
    """
    # Build the lemmatizer
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    # Run the lemmatizer on each word after splitting the input string, store results in the 'results' list
    results = []
    for word in string.split():
        results.append(lemmatizer.lemmatize(word))
    
    # Convert results back into a string
    string = ' '.join(results)
    
    # Return the resulting string
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
    
    # Return the resulting string
    return string


####################### PREP READMES FUNCTION ############################

def prep_readmes(df, col="readme_content"):
    """
    Takes in the dataframe and the column name that contains the corpus data, creates a column of cleaned data, then uses that 
    to create a column without stopwords that is lemmatized, performs a train-validate-test split, and returns train, validate,
    and test.
    """
    # Initialize a list to collect cleaned elements in the for-loop below
    cleaned_row = []
    
    # Iterate through the readme_content values...
    for i in df.readme_content.values:
        
        # Clean each value in the column and append to the 'cleaned_row' list
        cleaned_row.append(clean(i))
        
    # Assign the clean row content to a new column in the dataframe named 'cleaned_content
    df = df.assign(cleaned_content=cleaned_row)
    
    # Using a lambda, lemmatize all values in the 'cleaned_content' column and assign to a new column called 'lemmatized'
    df['lemmatized'] = df['cleaned_content'].apply(lambda x: lemmatize(remove_stopwords(x)))
    
    # Split the dataframe (70/15/15)
    train, validate, test = split_readmes(df)
    
    # Return train, validate, and test dataframes
    return train, validate, test
    
    
######################### WRANGLE READMES #################################
                               
def wrangle_readmes():
    """
    Acquires the readme data then preps it. Returns train, validate, and test dataframes
    """
    # Perform acquire and then prep the data, store in train, validate, and test dataframes
    train, validate, test = prep_readmes(acquire_readmes(), 'readme_content')
    
    # Return train, validate and test
    return train, validate, test


##################### IF NAME == MAIN STATEMENT ###########################

# Check if the script is being run as the main program
if __name__ == "__main__":
 
    # Call the function to scrape GitHub data and store the result in the 'data' variable
    data = scrape_github_data()  # Make sure you have the 'scrape_github_data()' function defined

    # Write the 'data' dictionary to a JSON file named "data2.json" with pretty formatting (indentation)
    with open("data2.json", "w") as json_file:
        json.dump(data, json_file, indent=1)
    
    
    
###########################################################################    
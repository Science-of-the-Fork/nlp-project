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
    results.to_csv("python_data", mode= "w")
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
    results.to_csv("java_script_data", mode= "w")
    return results

if __name__ == "__main__":
    data = scrape_github_data()
    json.dump(data, open("data2.json", "w"), indent=1)
    
    
######################## ACQUIRE READMES FUNCTION #####################

def acquire_readmes():
    """
    Loads in the python and javascript readme data and concats them. Returns the concatted dataframe.
    """
    python_df = get_github_python_data()
    python_df = python_df.rename(columns={'readme_content':'original'})
    javascript_df = get_github_java_script_data()
    javascript_df = javascript_df.rename(columns={'readme_content':'original'})
    df = pd.concat([python_df, javascript_df], axis=1)
    return df
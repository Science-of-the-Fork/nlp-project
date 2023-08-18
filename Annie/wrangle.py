
import os
import json
from typing import Dict, List, Optional, Union, cast
import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep

from env import github_token, github_username



# TODO: Make a github personal access token.
#     1. Go here and generate a personal access token: https://github.com/settings/tokens
#        You do _not_ need select any scopes, i.e. leave all the checkboxes unchecked
#     2. Save it in your env.py file under the variable `github_token`
# TODO: Add your github username to your env.py file under the variable `github_username`
# TODO: Add more repositories to the `REPOS` list below.


#----------ACQUIRE----------------------------
headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}

if headers["Authorization"] == "token " or headers["User-Agent"] == "":
    raise Exception(
        "You need to follow the instructions marked TODO in this script before trying to use it"
    )

#### FUNCTION TO GET REPO LINKS
def get_repo_links() -> List[str]:
    '''
    NOTE!!! VERY SLOW. IF DON'T HAVE A JSON FILE MAKE SURE TO RUN THIS FUNCTION AT LEAST FOR 1 HR
    
    Scraps the links of the repositories and saves them to the list
    '''
    filename = 'README_REPOS.json'
    REPOS=[]
    #headers = {"Authorization": f"token {github_token}", "User-Agent": github_username}
    languages = ['JavaScript', 'Python']
    # if the json file is available
    if os.path.isfile(filename):
        # read from json file
        with open(filename, "r") as json_file:
            REPOS = json.load(json_file)
    else:
        for i in range(1, 101):
            print(i)
            if i == 1:
                start_link = 'https://github.com/search?o=desc&q=stars:%3E1&s=forks&type=Repositories'
            else:
                start_link = f'https://github.com/search?o=desc&q=stars:%3E1&s=forks&type=Repositories'

            response = requests.get(start_link, headers=headers)
            if response.status_code != 200:
                print('problem' + str(response.status_code))
                sleep(20)
                response = requests.get(start_link, headers=headers)
            print(response.status_code)
            soup = BeautifulSoup(response.content, 'html.parser')

            all_blocks = soup.find_all('li', class_='repo-list-item hx_hit-repo d-flex flex-justify-start py-4 public source')
            if type(all_blocks) == None:
                print('all blocks fail')
                sleep(30)
                all_blocks = soup.find_all('li', class_='repo-list-item hx_hit-repo d-flex flex-justify-start py-4 public source')
            for block in all_blocks:
                try:
                    language = block.find('span', itemprop='programmingLanguage').text
                except:
                    continue
                if language in languages:
                    link = block.find('a', class_='v-align-middle')['href'][1:]
                    REPOS.append(link)
            sleep(20)
        
        with open(filename, "w") as outfile:
            json.dump(REPOS, outfile)
    return REPOS




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
        if "language" not in repo_info:
            raise Exception(
                "'language' key not round in response\n{}".format(json.dumps(repo_info))
            )
        return repo_info["language"]
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
    readme_download_url = get_readme_download_url(contents)
    if readme_download_url == "":
        readme_contents = ""
    else:
        readme_contents = requests.get(readme_download_url).text
    return {
        "repo": repo,
        "language": get_repo_language(repo),
        "readme_contents": readme_contents,
    }


# def scrape_github_data() -> List[Dict[str, str]]:
#     """
#     WARNING!!! VERY SLOW. IF DON'T HAVE A JSON FILE MAKE SURE TO RUN THIS FUNCTION AT LEAST FOR 1 HR

#     Loop through all of the repos and process them. Returns the processed data.
#     """
#     if os.path.isfile('data.json'):
#         # read from json file
#         with open('data.json', "r") as json_file:
#             data = json.load(json_file)
#     else:
#         REPOS = get_repo_links()
#         data = [process_repo(repo) for repo in REPOS]
#         with open('data.json', "w") as outfile:
#             json.dump(data, outfile)
#     return data



def get_clean_df() -> pd.DataFrame:
    '''
    Acquires the data from acquire helper file, saves it into a data frame.
    Cleans columns by appying cleaning functions from this file.
    Return:
        df: pd.DataFrame -> cleaned data frame
    '''

    # acquire a data from inshorts.com website
    df = pd.DataFrame(acquire.scrape_github_data())
    # news_df transformations
    # rename columns
    df.rename({'readme_contents':'original'}, axis=1, inplace=True)
    # create a column 'first_clean' hlml and markdown removed
    df['first_clean'] = df.original.apply(clean_html_markdown)
    # create a column 'clean' lower case, ascii, no stopwords
    df['clean'] = df.first_clean.apply(basic_clean).apply(tokenize).apply(remove_stopwords,extra_words=["'", 'space'])
    # only stems
    #df['stemmed'] = news_df.clean.apply(stem)
    # only lemmas
    df['lemmatized'] = df.clean.apply(lemmatize)
    # ENGINEER FEATURES BASED ON THE CLEAN TEXT COLUMN
    sia = nltk.sentiment.SentimentIntensityAnalyzer()
    # adds counpound sentiment score
    df['sentiment'] = df['clean'].apply(lambda doc: sia.polarity_scores(doc)['compound'])
    # numerical
    df['lemm_len'] = df.lemmatized.str.len()
    df['original_length'] = df.original.str.len()
    df['clean_length'] = df.clean.str.len()
    df['length_diff'] = df.original_length - df.clean_length
    # categorical
    df['has_#9'] = np.where(df.clean.str.contains('&#9;'), 1, 0)
    df['has_parts'] = np.where((df.clean.str.contains(' part ')) | (df.clean.str.contains('parts')), 1, 0)
    df['has_fix'] = np.where(df.clean.str.contains(' fix '), 1, 0)
    df['has_tab'] = np.where(df.clean.str.contains(' tab '), 1, 0)
    df['has_x'] = np.where(df.clean.str.contains(' x '), 1, 0)
    df['has_v'] = np.where(df.clean.str.contains(' v '), 1, 0)
    df['has_codeblock'] = np.where(df.clean.str.contains('codeblock'), 1, 0)
    df['has_image'] = np.where(df.clean.str.contains('image'), 1, 0)
    # change language to category
    df.language = pd.Categorical(df.language)
    # drop repo column
    df.drop('repo', axis=1, inplace=True)
    # drop 'clean_length' columns, as it is part of length_diff column
    df.drop('clean_length', axis=1, inplace=True)
    # reorder columns
    new_order = ['original', 'clean', 'lemmatized', 'lemm_len',
        'original_length', 'length_diff', 'has_#9', 'has_tab',\
        'has_parts', 'has_fix', 'has_x', 'has_v',\
       'has_codeblock', 'has_image', 'language']
    df = df[new_order]
    return df

####### PREPARATIONS FOR THE MODELING

def scale_numeric_data(X_train, X_validate, X_test):
    '''
    Scales numerical columns.
    Parameters:
        train, validate, test data sets
    Returns:
    train, validate, test data sets with scaled data
    '''
    # features to scale
    to_scale = ['sentiment', 'lem_length', 'original_length',  'length_diff']
    # create a scaler
    sc = MinMaxScaler()
    sc.fit(X_train[to_scale])
    # transform data
    X_train[to_scale] = sc.transform(X_train[to_scale])
    X_validate[to_scale] = sc.transform(X_validate[to_scale])
    X_test[to_scale] = sc.transform(X_test[to_scale])
    
    return X_train, X_validate, X_test

####### SPLITTING FUNCTIONS
def split_3(df):
    '''
    This function takes in a dataframe and splits it into 3 data sets
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    explore_columns = ['original', 'first_clean', 'clean', 'lemmatized', 'sentiment', 'lem_length',\
        'original_length', 'length_diff', 'language']
    df = df[explore_columns]
    #split_db class verision with random seed
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed, stratify=train_validate[target])
    return train, validate, test
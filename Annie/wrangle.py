
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

#-------------------PREPARE---------




def nlp_clean(text, extra_words=None, exclude_words=None):
    ''' This function does a basic clean Lowercased Tokenized Text with Latin Characters Only
    call to add cleaned column from readme_content'''
    # Basic clean
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-zA-Z0-9\s\']', '', text)
    
    # Tokenize
    filtered_text = text.split()
    
    return filtered_text

# Then the nlp_clean function to create a new column 'cleaned'
language_df['cleaned'] = language_df['readme_content'].apply(nlp_clean)




def lemmatize_text(text, extra_words=None, exclude_words=None):
    '''This function cleans, tokenizes, lemmatizes, and removes stop words. 
    Call to add 'lemmatized' column from 'original' column'''
    
    # Basic clean
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-zA-Z0-9\s\']', '', text)
    
    # Tokenize
    words = text.split()
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_text = ' '.join(lemmatized_words)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    if extra_words:
        stop_words.update(extra_words)
    if exclude_words:
        stop_words.difference_update(exclude_words)
    filtered_words = [word for word in lemmatized_words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    
    return filtered_text

# Example usage on a DataFrame's "readme_content" column
language_df['lemmatized'] = language_df['original'].apply(lemmatize_text)

    
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

def prep_process_csv(language_df):
    '''This function drops unneeded columns and creates a csv for processed language_to explore
        with based on ReadMe'''
    columns_to_drop = ['url', 'readme_content', 'repo_name']
    language_df.drop(columns=columns_to_drop, inplace=True)
    
    # Save the modified DataFrame to a new CSV file
    language_df.to_csv('processed_language_df.csv', index=False)
prep_process_csv(language_df)


#---------------Q1-------
def Q1_pie(language_df):
    ''' Function for Python vs JavaScript Pie'''
    # Calculate the value counts for each language
    language_counts = language_df['language'].value_counts()
    
    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(language_counts, labels=language_counts.index, autopct='%1.1f%%', colors=['green', 'yellow'])
    plt.title('Distribution of Python vs JavaScript Languages')
    plt.show()
    
def Q1_bar(language_df):
    ''' TOP 10 Python JavaScript word frequencies'''
    # Split each set of words by spaces, turn into a list, and calculate value counts
    Py_words_original = language_df[language_df['language'] == 'Python']['original'].str.split().explode().value_counts()
    Py_words_cleaned = language_df[language_df['language'] == 'Python']['cleaned'].explode().value_counts()
    Py_words_lemmatized = language_df[language_df['language'] == 'Python']['lemmatized'].str.split().explode().value_counts()
    
    Js_words_original = language_df[language_df['language'] == 'JavaScript']['original'].str.split().explode().value_counts()
    Js_words_cleaned = language_df[language_df['language'] == 'JavaScript']['cleaned'].explode().value_counts()
    Js_words_lemmatized = language_df[language_df['language'] == 'JavaScript']['lemmatized'].str.split().explode().value_counts()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 3, 1)
    Py_words_original[:10].plot(kind='bar', color='blue', alpha=0.7)
    plt.title("Top 10 Python Original Word Frequencies")
    
    plt.subplot(2, 3, 2)
    Py_words_cleaned[:10].plot(kind='bar', color='green', alpha=0.7)
    plt.title("Top 10 Python Cleaned Word Frequencies")
    
    plt.subplot(2, 3, 3)
    Py_words_lemmatized[:10].plot(kind='bar', color='orange', alpha=0.7)
    plt.title("Top 10 Python Lemmatized Word Frequencies")
    
    plt.subplot(2, 3, 4)
    Js_words_original[:10].plot(kind='bar', color='blue', alpha=0.7)
    plt.title("Top 10 JavaScript Original Word Frequencies")
    
    plt.subplot(2, 3, 5)
    Js_words_cleaned[:10].plot(kind='bar', color='green', alpha=0.7)
    plt.title("Top 10 JavaScript Cleaned Word Frequencies")
    
    plt.subplot(2, 3, 6)
    Js_words_lemmatized[:10].plot(kind='bar', color='orange', alpha=0.7)
    plt.title("Top 10 JavaScript Lemmatized Word Frequencies")
    
    plt.tight_layout()
    plt.show()
    
def Q1_stat(language_df):
    alpha = 0.05
    
    # Create a contingency table with the counts of Python and JavaScript languages
    language_table = pd.crosstab(language_df['language'], columns='count')
    
    # Perform the Chi-squared test of independence
    chi2, p_value, dof, expected = chi2_contingency(language_table)
    
    # Print the results
    print(f"Chi-squared value: {chi2}")
    print(f'p_value = {p_value:.4f}')
    print(f"Degrees of freedom: {dof}")
    print("Expected frequencies:")
    print(expected)
    print('\n----')
    print(f'p_value = {p_value:.4f}')
    print(f'The p-value is less than the alpha: {p_value < alpha}')
    if p_value < alpha:
        print('We reject the null')
    else:
        print("we fail to reject the null")



#----------------- Q2------
def top_10_words(language_df):

    # Split each set of words by spaces, turn into a list, and calculate value counts
    original_words = language_df['original'].str.split().explode().value_counts()
    cleaned_words = language_df['cleaned'].explode().value_counts()
    lemmatized_words = language_df['lemmatized'].str.split().explode().value_counts()
    
    # Plotting
    plt.figure(figsize=(10, 10))
    
    original_plot = original_words[:10].plot(kind='bar', color='blue', alpha=0.7, label='Original')
    cleaned_plot = cleaned_words[:10].plot(kind='bar', color='green', alpha=0.7, label='Cleaned')
    lemmatized_plot = lemmatized_words[:10].plot(kind='bar', color='orange', alpha=0.7, label='Lemmatized')
    
    # Annotate bars with their counts
    for p in original_plot.patches:
        original_plot.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    
    for p in cleaned_plot.patches:
        cleaned_plot.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    
    for p in lemmatized_plot.patches:
        lemmatized_plot.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    
    plt.title("Top 10 Word Frequencies")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.show()
    
#------------Q3--------------------------

def Q3_PyBigram_barplot(language_df):
    # Lemmatized column
    Py_lemmatized = language_df[language_df['language'] == 'Python']['lemmatized'].str.split().explode()
    #Top 20 Bigrams
    top_20_python_bigrams = (pd.Series(nltk.ngrams(Py_lemmatized, 2))
                          .value_counts()
                          .head(20))
    
    # Sort the bigrams in descending order by frequency
    top_20_python_bigrams = top_20_python_bigrams.sort_values(ascending=False)
    
    # Create the horizontal bar plot
    plt.figure(figsize=(12, 8))
    bars = top_20_python_bigrams.plot(kind='barh', color = '#008B8B', width=0.9)
    
    # Add count annotations at the end of the bars
    for bar in bars.patches:
        plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, 
                 f'{int(bar.get_width())}', 
                 va='center', ha='left', fontsize=10, color='black')
    
    plt.title('Top 20 Most Frequently Occurring Python Bigrams')
    plt.xlabel('Number Occurrences')
    plt.ylabel('Bigram')
    
    # Make the labels pretty
    ticks, labels = plt.yticks()
    new_labels = [f'{bigram[0]} {bigram[1]}' for bigram in top_20_python_bigrams.index]
    plt.yticks(ticks, new_labels)
    plt.show()

#------------Q4--------------------

def Q4_PyTrigram_barplot(language_df):
    
    top_20_python_trigrams = (pd.Series(nltk.ngrams(Py_lemmatized, 3))
                          .value_counts()
                          .head(20))
    
    # Sort the Trigram in descending order by frequency
    top_20_python_trigrams = top_20_python_trigrams.sort_values(ascending=False)
    
    # Create the horizontal bar plot
    plt.figure(figsize=(12, 8))
    bars = top_20_python_trigrams.plot(kind='barh', color='#008B8B', width=0.9)
    
    # Add count annotations at the end of the bars
    for bar in bars.patches:
        plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, 
                 f'{int(bar.get_width())}', 
                 va='center', ha='left', fontsize=10, color='black')
    
    plt.title('Top 20 Most Frequently Occurring Python Trigrams')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Trigram')
    
    ticks, labels = plt.yticks()
    new_labels = [f'{trigram[0]} {trigram[1]} {trigram[2]}' for trigram in top_20_python_trigrams.index]
    plt.yticks(ticks, new_labels)
    
    plt.tight_layout()
    plt.show()
        
#--------------Q5----------
def Q5_JsBigram_barplot(language_df):
    # Original column
    Js_original = language_df[language_df['language'] == 'JavaScript']['original'].str.split().explode()
    
    # Cleaned column
    Js_cleaned = language_df[language_df['language'] == 'JavaScript']['cleaned'].explode()
    
    # Lemmatized column
    Js_lemmatized = language_df[language_df['language'] == 'JavaScript']['lemmatized'].str.split().explode()
    
    top_20_javascript_bigrams = (pd.Series(nltk.ngrams(Js_lemmatized, 2))
                          .value_counts()
                          .head(20))
    
    # Sort the bigrams in descending order by frequency
    top_20_javascript_bigrams = top_20_javascript_bigrams.sort_values(ascending=False)
    
    # Create the horizontal bar plot
    plt.figure(figsize=(12, 8))
    bars = top_20_javascript_bigrams.plot(kind='barh', color = '#008B8B', width=0.9)
    
    # Add count annotations at the end of the bars
    for bar in bars.patches:
        plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, 
                 f'{int(bar.get_width())}', 
                 va='center', ha='left', fontsize=10, color='black')
    
    plt.title('Top 20 Most Frequently Occurring JavaScript Bigrams')
    plt.xlabel('Number Occurrences')
    plt.ylabel('Bigram')
    
    ticks, labels = plt.yticks()
    new_labels = [f'{bigram[0]} {bigram[1]}' for bigram in top_20_javascript_bigrams.index]
    plt.yticks(ticks, new_labels)
    
    plt.show()
#-----------------Q6----------------
def Q6_JsTrigram_barplot(language_df):
    top_20_javascript_trigrams = (pd.Series(nltk.ngrams(Js_lemmatized, 3))
                          .value_counts()
                          .head(20))
    
    top_20_javascript_trigrams.head()
    # Sort the bigrams in descending order by frequency
    top_20_javascript_trigrams = top_20_javascript_trigrams.sort_values(ascending=False)
    
    # Create the horizontal bar plot
    plt.figure(figsize=(12, 8))
    bars = top_20_javascript_trigrams.plot(kind='barh', color='#008B8B', width=0.9)
    
    # Add count annotations at the end of the bars
    for bar in bars.patches:
        plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2, 
                 f'{int(bar.get_width())}', 
                 va='center', ha='left', fontsize=10, color='black')
    
    plt.title('Top 20 Most Frequently Occurring JavaScript Trigrams')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Trigram')
    
    ticks, labels = plt.yticks()
    new_labels = [f'{trigram[0]} {trigram[1]} {trigram[2]}' for trigram in top_20_javascript_trigrams.index]
    plt.yticks(ticks, new_labels)
    
    plt.tight_layout()
    plt.show()

#----------TF-IDF-----------
def tf_idf():
    '''
    This function takes in a dataframe and splits it into 3 data sets
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the  dataset. Uses lemmatized only for TF-IDF
    The function returns, in this order, train, validate and test dataframes. 
    '''
    # Load and preprocess data (replace with your own preprocessing steps)
    data = language_df  
    
    # Create feature vectors
    tf_vectorizer = CountVectorizer(max_features=1000)  # You can adjust max_features
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features
    
    X_tf = tf_vectorizer.fit_transform(data['lemmatized']).toarray()
    X_tfidf = tfidf_vectorizer.fit_transform(data['lemmatized']).toarray()
    y = data['language']
    
    # Split the data
    X_train_tf, X_test_tf, y_train, y_test = train_test_split(X_tf, y, test_size=0.2, random_state=123)
    X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.3, random_state=123)
    
    # Build and train models
    model_tf = LogisticRegression()
    model_tfidf = LogisticRegression()
    
    model_tf.fit(X_train_tf, y_train)
    model_tfidf.fit(X_train_tfidf, y_train)
    
    # Evaluate performance
    y_pred_tf = model_tf.predict(X_test_tf)
    y_pred_tfidf = model_tfidf.predict(X_test_tfidf)
    
    accuracy_tf = accuracy_score(y_test, y_pred_tf)
    accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
    
    print("TF Model Accuracy:", accuracy_tf)
    print("TF-IDF Model Accuracy:", accuracy_tfidf)
    
    print("TF Model Classification Report:")
    print(classification_report(y_test, y_pred_tf))
    
    print("TF-IDF Model Classification Report:")
    print(classification_report(y_test, y_pred_tfidf))
    
    
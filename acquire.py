import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_and_save_articles(urls, csv_filename):
   
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    else:
        df = pd.DataFrame(columns=['title', 'content'])
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    
    for url in urls:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        title = soup.find('h1').text.strip()
        content = ' '.join([p.text.strip() for p in soup.find_all('p')])
        
        article = {
            'title': title,
            'content': content
        }
        
        
        df = df.append(article, ignore_index=True)
    df.to_csv(csv_filename, index=False)
    print(f"Articles saved to '{csv_filename}'")
    
    return df




CSV_FILENAME = "inshorts_articles.csv"

def scrape_inshorts_articles(urls):
    articles = []

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        article_cards= soup.find_all(itemtype='http://schema.org/NewsArticle')

        for card in article_cards:
            headline = card.find('span', itemprop='headline').text
            summary = card.find('div', itemprop='articleBody').text
            author = card.find('span', class_ = 'author').text
            date = card.find('span', itemprop = 'datePublished').text
            day = card.find('span', clas='date').text
            
            category = url.split('/')[-1]
            
            
            articles.append({
                'headline': headline,
                'summary': summary,
                'author': author,
                'time': date,
                'day': day,
                'category': category
                
            })

    df = pd.DataFrame(articles)
    return df

def get_inshorts_data():
    if os.path.exists(CSV_FILENAME):
        print("Loading data from CSV...")
        return pd.read_csv(CSV_FILENAME)
    else:
        base_url = "https://inshorts.com/en/read/"
        pages = [
            "business",
            "sports",
            "technology",
            "entertainment",
            "india",
            "politics",
            "startups",
            "hatke",
            "international",
            "automobile",
            "science",
            "travel",
            "miscellaneous",
            "fashion",
            "education",    
        ]      

        urls = [base_url + page for page in pages]
        result_df = scrape_inshorts_articles(urls)
        result_df.to_csv(CSV_FILENAME, index=False)
        print("Data scraped and saved to CSV.")
        return result_df


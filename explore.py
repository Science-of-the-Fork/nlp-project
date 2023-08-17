import re
import unicodedata
import pandas as pd
import nltk
from wordcloud import WordCloud


def generate_trigrams(text):
    words = text
    trigrams = []

    if len(words) < 3:
        return trigrams

    for i in range(len(words) - 2):
        trigram = " ".join(words[i:i+3])
        trigrams.append(trigram)

    return trigrams

def generate_bigrams(text):
    words = text
    bigrams = []

    if len(words) < 2:
        return bigrams

    for i in range(len(words) - 1):
        bigram = " ".join(words[i:i+2])
        bigrams.append(bigram)

    return bigrams

#BARPLOT CODE EX.
#top_spam_freq = spam_freq.head()
#top_spam_freq.plot(kind='bar', title='Top Spam Bigram Frequencies')
#plt.xlabel('Bigram')
#plt.ylabel('Frequency')
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()
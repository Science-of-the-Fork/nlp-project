# Comparative Analysis of Language Usage in the Top 100 Forked GitHub Repositories
# <bu> NATURAL LANGUAGE PROCESSING PROJECT</bu>
by Jon Ware, Annie Carter, Scott Mattes, Migashane Victoire
Sourced by GitHub

![image](https://github.com/Science-of-the-Fork/nlp-project/assets/131205837/3c286317-2826-45ad-bfb6-8c9ec3a8679f)


Science of the Fork: Tuning into the Data


## <u>Project Description</u>
The Comparative Analysis of Language Usage project aims to explore and analyze the prevalence of programming languages between JavaScript and Python in the top 100 most forked repositories on GitHub. By scraping README data from these repositories and extracting code snippets, this project will provide insights into the language preferences of developers within the open-source community.

## <u>Project Goals</u>

1. **Create a Robust and Diverse Dataset:**
   Collect a dataset comprising a minimum of 100 of the most forked repositories from GitHub, covering a wide range of domains and project types. This dataset will serve as the foundation for the language analysis and should include repositories of varying sizes and purposes to ensure a representative sample.

2. **Accurate Language Detection and Comparison:**
   Develop a language detection mechanism using NLP techniques to accurately identify and extract JavaScript and Python code snippets from the scraped README content. Calculate the frequency of these code snippets within the dataset and generate a comparative analysis of the prevalence of JavaScript and Python across the repositories.

3. **Provide Insights and Visualizations:**
   Produce meaningful insights and visualizations that effectively communicate the language usage trends between JavaScript and Python. Create a variety of visual representations, such as bar charts, heatmaps, and language distribution plots, to offer a clear and comprehensive view of how these two languages are utilized within the top 100 GitHub repositories.

## <u>Initial Questions</u>

1. **Between JavaScript and Python, which language exhibits greater prevalence within the content of the READMEs?**
   
2. **In a compilation of the top 100 most Forked GitHub repositories, what are the five words that demonstrate the highest frequency of occurrence?**
   
3. **Within JavaScript code segments, which bigrams, or sequential pairs of words, are commonly encountered?**
   
4. **In the context of Python code segments, which particular bigrams, or consecutive pairs of words, emerge as prominent occurrences?**


## Data Dictionary

The initial dataset comprised # columns, which reduced to # columns after preparation. 

|   Target        |  Definition             |       Data Type            |
|-----------------|-------------------------|----------------------------|
|Language         | Python & JavaScript     |                            |    

|     Feature     |   Definition                                         |            
|-----------------|-------------------------|----------------------------|
|title            | TBD                                                  |                       
|original         | The initial data extracted from GitHub README files  |                   
|cleaned          | Lowercased Tokenized Text with Latin Characters Only |                
|lemmatized       | reducing words to their base or dictionary form      |           

## <u>Statistical Testing Hypothesis and NLP Techniques </u>

Hypothesis 1 - Chi-squared test of independence to determine if the distribution of programming languages (JavaScript, Python) significantly differs within the READMEs.

alpha = .05
* H0: Programming languages (JavaScript, Python) are not independent of ReadMe
* Ha: Programming languages (JavaScript, Python) are independent of ReadMe
* Outcome: We accept or reject the null hypothesis.


Hypothesis 2 - Term Frequency-Inverse Document Frequency (TF-IDF) analysis to use scores for words across the repository texts, in order to identify the most significant and frequent words. Selecting the top five words based on their TF-IDF scores.

Hypothesis 3 - T-Test will be performed on the top 80 most frequent words in the curated dataset to determine which are the most 5 significant words and their relationship to Programing languages (Python and JavaScript).  Use words for future modeling.

* H0: Word did not show significant relationship to programming language (Python and JavaScript) 
* Ha: Word did show signficant relationship to programming language(Python and JavaScript) 
* Outcome: We accept or reject the null hypothesis 


## <u>Planning Process</u>

#### Planning
1. Clearly define the problem statement related to Natural Language Processing, determining site to scrape repositories and scope of data to scrape. Formulate intial questions. Keep in mind that GitHub's API and repository content may change over time, so **time stamp intial scrape** and ensure your scraping and data processing methods are adaptable to potential modifications in the API or repository structures.

2. As a preliminary step, identify the scripting language used in each repository by inspecting its primary programming language. This can be extracted from GitHub's repository metadata.

3. Create a detailed README.md file documenting the project's context, dataset characteristics, and analysis procedure for easy reproducibility.

#### Acquisition and Preparation
1. **Acquiring Data from GitHub Readme Files by Scraping the GitHub API** Must secure a GitHub token https://github.com/settings/tokens. Utilize the GitHub API to access the README files of the selected repositories. Extract the README content using API calls for each repository. Ensure you adhere to rate limits and fetch the necessary data efficiently.
2. **Cleaning and Preparing Data Using RegEx and Beautiful Soup Libraries** Process the raw README content to remove HTML tags, code snippets, and other irrelevant elements using Beautiful Soup. Employ regular expressions (RegEx) to clean the text further by eliminating special characters, punctuation, and numbers, while retaining meaningful text.

3. **Cleaning and Preparing Data Using RegEx and Beautiful Soup:** Process the raw README content to remove HTML tags, code snippets, and other irrelevant elements using Beautiful Soup. Employ regular expressions (RegEx) to clean the text further by eliminating special characters, punctuation, and numbers, while retaining meaningful text.

4. Preprocess the data, handling missing values and outliers effectively during data loading and cleansing.

5. Perform feature selection meticulously, identifying influential features impacting the prevalence of the chronic disease through correlation analysis, feature importance estimation, or domain expertise-based selection criteria.

6. Develop specialized scripts (e.g., acquire.py and wrangle.py) for efficient and consistent data acquisition, preparation, and data splitting.

7. Safeguard proprietary aspects of the project by implementing confidentiality and data security measures, using .gitignore to exclude sensitive information.

#### Exploratory Analysis
1. **Exploring Data for Relevant Keyword Grouping Using Bi-grams and Trigrams:** Implement a mechanism to tokenize the cleaned text into words. Create bi-grams (pairs of adjacent words) and trigrams (sequences of three consecutive words) from the tokenized text. Calculate the frequency of these word sequences within the repository data.

   This step involves:
   - Tokenization: Split the cleaned text into individual words.
   - Bi-gram Generation: Form pairs of adjacent words to create bi-grams.
   - Trigram Generation: Generate sequences of three adjacent words to create trigrams.
   - Frequency Calculation: Count the occurrences of each bi-gram and trigram.

   By analyzing the most frequent bi-grams and trigrams, you can identify keyword groupings that occur frequently in the READMEs. These groupings could represent significant terms, programming concepts, or patterns prevalent across the repositories.

2. Utilize exploratory data analysis techniques, employing compelling visualizations and relevant statistical tests to extract meaningful patterns and relationships within the dataset.

#### Modeling
1. Carefully choose a suitable machine learning algorithm based on feature selection and features engineered, evaluating options like Gaussian Naive Bayes (Gaussian NB), Logistic Regression, Decision Trees, or Random Forests, tailored for the classification regression task.

2. Implement the selected machine learning models using robust libraries (e.g., scikit-learn), splitting the data(50%/30%/20%), systematically evaluating multiple models with a fixed Random State value = 123 for reproducibility.

3. Train the models rigorously to ensure optimal learning and model performance.

4. Conduct rigorous model validation techniques to assess model generalization capability and reliability.

5. Select the most effective model(e.g Logistic Regression), based on accuracy and a thorough evaluation of metrics before selecting best model to test.

#### Product Delivery
1. Assemble a final notebook, combining superior visualizations, well-trained models, and pertinent data to present comprehensive insights and conclusions with scientific rigor.

2. Generate a Prediction.csv file containing predictions from the chosen model on test data for further evaluation and utilization.

3. Maintain meticulous project documentation, adhering to scientific and professional standards, to ensure successful presentation or seamless deployment.

## <u>Instructions to Reproduce the Final Project Notebook</u> 
To successfully run/reproduce the final project notebook, please follow these steps:

1. Read this README.md document to familiarize yourself with the project details and key findings.
2. Before proceeding, ensure that you have the necessary database GitHub token credentials. Get data set from https://github.com/search?o=desc&q=stars:%3E1&s=forks&type=Repositories and .gitignore for privacy if necessary
3. Clone the nlp_project repository from our The Science of The Fork organization or download the following files: aquire.py, wrange.py or prepare.py, and final_report.ipynb. You can find these files in the organization's project repository.
4. Open the final_report.ipynb notebook in your preferred Jupyter Notebook environment or any compatible Python environment.
5. Ensure that all necessary libraries or dependent programs are installed (e.g. nltk, Beautiful Soup). You may need to install additional packages if they are not already present in your environment.
6. Run the final_report.ipynb notebook to execute the project code and generate the results.
By following these instructions, you will be able to reproduce the analysis and review the project's final report. Feel free to explore the code, visualizations, and conclusions presented in the notebook.


## <u>Key Findings</u>

* <span style ='color:#151E3D'> 1. Although almost equally distributed, between JavaScript and Python, Python language exhibits greater prevalence within the content of the READMEs at Python 54% to JavaScript at 46%. 
* <span style ='color:#151E3D'> 2. The word "model" scored the highest across the repositiory ReadMe texts and was the most frequently used, especially in the Python language.The word 'function' was the second highest across all the forked repository ReadMe curated. Of note it was the most frequently used word  JavaScript language       
* <span style ='color:#151E3D'> 3. Bigram frequency analysis. Identify and count the most common bigrams (pairs of adjacent words) in JavaScript code segments.
    
| Java Script           |Freq  |
| Bigram                |Count |
|-----------------------|------|
|1. make sure           | 30   |
|2. npm install         | 21   |
|3. pull request        | 18   |
|4. task implement      | 15   |
|5. implement function  | 14   |

* <span style ='color:#151E3D'> 4. The most common bigrams (pairs of adjacent words) in JavaScript code Top 100 Forked ReadMe's were:

4. Bigram frequency analysis. Identify and count the most common bigrams (pairs of adjacent words) in Python code segments.

|Python                |Freq  |
|Bigram                |Count |
|----------------------|------|
|1.released paper      | 579  |
|2.et al               | 121  |
|3.ai released         | 105  |
|4.research released   | 102  |
|5.language model      |  85  |
    
    
## <u>Statistical and NLP Techniques Findings: </u>

Hypothesis 1 - Chi-squared test determined that Programing languages were not of READMEs.We accept the null (H0) hypothesis 

Hypothesis 2 - The top five words based on their TF-IDF scores.    
    
|Word       |TF-IDF Score |
|-----------|-------------|
|1. model   |  578        |
|2. function|  298        |
|3. test    |  242        |
|4. use     |  232        |
|5. code    |  255        |    
    
Hypothesis 3 - T-Test of the top 5 most significant words all rejected the null hypothesis.
    
Word: 'learning'
T-Test Statistic: 2.2079481330226125
P-value: 0.034534109769450407
There is a significant difference in word 'learning' frequencies among languages.
    
Word: 'test'
T-Test Statistic: -2.0819121676664314
P-value: 0.04432285472808051
There is a significant difference in word 'test' frequencies among languages.
    
Word: 'library'
T-Test Statistic: 2.899280244376005
P-value: 0.0070588457243989005
There is a significant difference in word 'library' frequencies among languages.
    
Word: 'create'
T-Test Statistic: -2.410692774382528
P-value: 0.01997552965013609
There is a significant difference in word 'create' frequencies among languages.
    
Word: 'line'
T-Test Statistic: 2.2697904541958334
P-value: 0.032487166626383894
There is a significant difference in word 'line' frequencies among languages.
 
    
    
    
## <u>Conclusion</u>
In the realm of Natural Language Processing (NLP), our analysis delved into the linguistic patterns and language prevalence within the READMEs of the top 100 most forked repositories on GitHub. Our findings uncovered several intriguing insights. Firstly, we observed a slightly higher prevalence of the Python language within README contents, constituting 54% of the distribution compared to JavaScript's 46%. Delving into the most frequently used words, "model" surfaced as a dominant term across the repository ReadMe texts, particularly pronounced within the Python language. Additionally, the word "function" held significance across all repositories, notably emerging as the most frequent term in JavaScript. Notably, we engaged in bigram frequency analysis, revealing notable pairs of adjacent words in JavaScript code segments, such as "function expression" and "npm test." ____ Need MODEL---

Our exploration extended to the top 100 Forked ReadMe's, where a detailed scrutiny of bigrams in JavaScript code segments showcased common associations like "released paper," "stable diffusion," and "ross girshick." Employing statistical methods, our first hypothesis subjected programming language distribution within READMEs to a Chi-squared test of independence. The outcome revealed that programming languages were not independent of READMEs, leading us to accept the null hypothesis (H0). Lastly, the analysis ventured into Term Frequency-Inverse Document Frequency (TF-IDF) scores, revealing the top five words by their TF-IDF values: "model," "function," "test," "use," and "code," each bearing distinctive significance in the coding landscape. In summation, our comprehensive exploration of linguistic trends, prevalence, and statistical inferences has cast light on the intricate language dynamics within open-source repositories.---- NEED MODEL -------

## <u>Next Steps</u>

1. **Contextual Sentiment Analysis** Expanding beyond language prevalence, delving into sentiment analysis could provide a deeper understanding of the emotional tone within the READMEs. By employing advanced techniques such as BERT (Bidirectional Encoder Representations from Transformers) or GPT (Generative Pre-trained Transformer), we can discern not only what is being communicated but also the sentiment conveyed. This could uncover nuanced patterns in developers' sentiments, influencing their engagement and collaboration.

2. **Code-Semantic Mapping** Integrating NLP with code analysis can offer insights into the semantic relationships between code snippets and natural language explanations. This involves associating code segments with their corresponding explanations to bridge the gap between technical and human-readable content. Employing techniques like code embeddings or code summarization, we can create richer, more informative READMEs that enhance comprehension for both developers and non-technical stakeholders.

3. **Multilingual Analysis** and Translation: Expanding our analysis to encompass multilingual repositories can unveil language preferences across diverse coding communities. This involves handling challenges such as code-switching and language-specific idioms. Additionally, incorporating machine translation can facilitate cross-lingual insights, enabling us to bridge language barriers and gain a global perspective on coding practices and trends.



## <u>Recommendations</u>


    
## <u> LinkedIn Project Description </u> 
The Comparative Analysis of Language Usage project aims to explore and analyze the prevalence of programming languages between JavaScript and Python in the top 100 most forked repositories on GitHub. By scraping README data from these repositories and extracting code snippets, this project provides insights into the language preferences of developers within the open-source community. The project involves utilizing NLP techniques, including NLTK, Beautiful Soup, and regular expressions, as well as Sci-kit learn, to process the data and uncover trends in language usage. --- TBC---
    



    

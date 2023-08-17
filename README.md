# Comparative Analysis of Language Usage in the Top 100 Forked GitHub Repositories
# <bu> NATURAL LANGUAGE PROCESSING PROJECT</bu>
by Jon Ware, Annie Carter, Scott Mattes, Migashane Victoire
Sourced by GitHub

![image](https://github.com/Science-of-the-Fork/nlp-project/assets/131205837/3c286317-2826-45ad-bfb6-8c9ec3a8679f)


Science of the Fork: Tunning into the Data


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
   
5. **Within JavaScript code samples, what are the frequently occurring trigrams, or sequences of three words?**
   
6. **In Python code excerpts, what trigrams, or sets of three consecutive words, stand out as prominent linguistic patterns?**

## Data Dictionary

The initial dataset comprised # columns, which reduced to # columns after preparation. 

| Original                    |   Target     |       Data Type           |       Definition             |
|-----------------------------|------------- |--------------------------|------------------------------ |
|Python and Java Script       |    Language  | ##### non-null  dtype   |      target variable           |



|     Original                |   Feature     |       Data Type          |     Definition               |
|-----------------------------|-------------- |------------------------ |------------------------------ |
|    TBD                      |TBD            | ##### non-null  dtype   | TBD                           |
|    TBD                      |TBD            | ##### non-null  dtype   | TBD                           |
|    TBD                      |TBD            | ##### non-null  dtype   | TBD                           | 


## <u>Statistical Testing Hypothesis and NLP Techniques </u>
Hypothesis 1 - Chi-squared test of independence to determine if the distribution of programming languages (JavaScript, Python) significantly differs within the READMEs.

alpha = .05
* H0 =  TBD
* Ha = TBD
* Outcome: We accept or reject the Null Hypothesis.

Hypothesis 2 - Term Frequency-Inverse Document Frequency (TF-IDF) analysis to use scoresfor words across the repository texts, in order to identify the most significant and frequent words. Selecting the top five words based on their TF-IDF scores.

|Word |TF-IDF Score |
|-----|-------------|
|1.   |             |
|2.   |             |
|3.   |             |
|4.   |             |
|5.   |             |
* Outcome: 

Hypothesis 3 -Bigram frequency analysis. Identify and count the most common bigrams (pairs of adjacent words) in JavaScript code segments
|Java Script|Freq  |
|Bigram     |Count |
|-----------|------|
|1.         |      |
|2.         |      |
|3.         |      |
|4.         |      |
|5.         |      |

* Outcome: 

Hypothesis 4 -Bigram frequency analysis. Identify and count the most common bigrams (pairs of adjacent words) in Python code segments
|Python|Freq  |
|Bigram     |Count |
|-----------|------|
|1.         |      |
|2.         |      |
|3.         |      |
|4.         |      |
|5.         |      |

* Outcome: 


## <u>Planning Process</u>

#### Planning
1. Clearly define the problem statement related to Natural Language Processing, determining site to scrape repositories and scope of data to scrape. Formulate intial questions. Keep in mind that GitHub's API and repository content may change over time, so **time stamp intial scrape** and ensure your scraping and data processing methods are adaptable to potential modifications in the API or repository structures.

2. As a preliminary step, identify the scripting language used in each repository by inspecting its primary programming language. This can be extracted from GitHub's repository metadata.

3. Create a detailed README.md file documenting the project's context, dataset characteristics, and analysis procedure for easy reproducibility.

#### Acquisition and Preparation
1. **Acquiring Data from GitHub Readme Files by Scraping the GitHub API** Most secure a GitHub token https://github.com/settings/tokens. Utilize the GitHub API to access the README files of the selected repositories. Extract the README content using API calls for each repository. Ensure you adhere to rate limits and fetch the necessary data efficiently.
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

* <span style ='color:#151E3D'> TBD
    
* <span style ='color:#3D3D3D'> TBD
    
    
## <u>Conclusion</u>
TBD

## <u>Next Steps</u>

1. TBD
2. TBD
3. TBD

## <u>Recommendations</u>


    
## <u> LinkedIn Project Description </u> 
The Comparative Analysis of Language Usage project aims to explore and analyze the prevalence of programming languages between JavaScript and Python in the top 100 most forked repositories on GitHub. By scraping README data from these repositories and extracting code snippets, this project provides insights into the language preferences of developers within the open-source community. The project involves utilizing NLP techniques, including NLTK, Beautiful Soup, and regular expressions, as well as Sci-kit learn, to process the data and uncover trends in language usage. --- TBC---
    



    

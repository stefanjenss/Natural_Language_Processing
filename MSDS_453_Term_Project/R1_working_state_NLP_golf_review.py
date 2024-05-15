# %%
"""
PRE-PHASE: SET UP THE ENVIRONMENT

Set up the environment with the necessary libraries, packages, and display configurations

Note: this code is written in Python 3.9, as this version of Python is compatable with the GenSim library
"""
# Enable multiple outputs in a single cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Import the necessary libraries and packages
# Regular expressions, string manipulation, and file system operations
import re, string, os
# Data manipulation and analysis
import pandas as pd
# Scientific computing
import numpy as np
from numpy import triu
import scipy
# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Natural language processing
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import silhouette_score
# Gensim for topic modeling
from gensim.models import Word2Vec, LdaModel, TfidfModel
from gensim import corpora
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Display configurations
plt.style.use('ggplot')
sns.set_style('whitegrid')
# Set the random seed for reproducibility
np.random.seed(42)

# %%
"""
PHASE 0: IMPORT THE GOLF COURSE REVIEW CORPUS AND CONVERSION TO CSV FILE AND DATAFRAME
    1. Firstly, since I created the golf course review corpus using Excel, I will need to convert the Excel file to a CSV file. 
       I will use the pandas library to do this.
    2. Secondly, I will use the pandas library to read the CSV file and create a DataFrame.
"""

# Convert the Excel file to a CSV file
# Read the Excel file
FILE = 'golf_course_review_corpus_V2.xlsx'
golf_course_review = pd.read_excel(FILE)

# Save the DataFrame to a CSV file
golf_course_review.to_csv('golf_course_review_corpus.csv', index=False)

# Read the CSV file and create a DataFrame
golf_course_review = pd.read_csv('golf_course_review_corpus.csv')
golf_course_review.head()

# %%
#### PHASE 1: DATA EXPLORATION ####
# NOT INCLUDED YET#

#%%
#### PHASE 2: DATA PREPROCESSING ####

# Ensure that the necessary libraries for this phase are imported
import spacy
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd

# Load necessary NLP models and stopwords that will be used for data preprocessing
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])   # Load the spaCy English language model, disable the parser and named entity recognition for efficiency
standard_stop_words = set(stopwords.words('english'))   # Load the English stopwords from NLTK
tokenizer = RegexpTokenizer(r'\w+')   # Define a regular expression tokenizer to remove and punctuation and perform tokenization using NLTK's RegexpTokenizer
domain_specific_stopwords = set([
    'course', 'play', 'hole', 'green', 'par', 'tee', 'yard', 'golf', 'one', 'bunker', 'fairway', 'leave', 'shot', 'right', 'good'
])

# Define a function to preprocess the text data
def preprocess_text(text):
    """
    Preprocessing a single document by applying several preprocessing steps:
        - Tokenization
        - Non-alphabetic token removal
        - Stort token removal
        - Lowercasing
        - Stopword removal
        - Lemmatization
        - Domain-specific stopword removal
    Args:
        text (str): The original text of the golf course review to be preprocessed
    Returns:
        str: The preprocessed text of the golf course review
    """
    tokens = tokenizer.tokenize(text.lower())   # Tokenize the text and convert it to lowercase
    tokens = [token for token in tokens if token.isalpha() and len(token) > 2]  # Remove non-alphabetic tokens and tokens with less than 3 characters
    tokens = [token for token in tokens if token not in standard_stop_words]   # Remove standard stopwords
    doc = nlp(' '.join(tokens))     # Lemmatization (Part 1): Convert the tokens back to a string for spaCY processing
    tokens = [token.lemma_ for token in doc if token.lemma_ not in domain_specific_stopwords]   # Lemmatization (Part 2): Lemmatize the tokens and remove domain-specific stopwords
    return ' '.join(tokens)   # Return the preprocessed text as a single string

# Apply the preprocessing function to the 'review_text' column of the golf_course_review DataFrame
golf_course_review['cleaned_review_text'] = golf_course_review['review_text'].apply(preprocess_text)

# Display the first few rows of the DataFrame with the cleaned review text
golf_course_review[['review_text', 'cleaned_review_text']].head()

"""
Improvements from previous version:
    - Efficiency: By loading the NLP model and other resources only once outside the function, the code avoids redundant operations, which can be a significant 
      performance enhancement when processing large datasets.
    - Refactoring: Combining similar operations reduces the complexity and increases the readability of the code.
"""

# %%
#### PHASE 3: FEATURE ENGINEERING ####
### Phase 3.1: Feature Extraction using TF-IDF Vectorization ###

# Import the necessary libraries for this phase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.metrics import silhouette_score

# Create a pipeline with TF-IDF Vectorizer and K-Means clustering
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('kmeans', KMeans(random_state=42))
])

# Define the parameter grid for GridSearchCV
param_grid = {
    'tfidf__max_df': [0.75, 0.85, 0.95],  # Maximum document frequency for the TF-IDF vectorizer
    'tfidf__min_df': [0.01, 0.05, 0.1],     # Minimum document frequency for the TF-IDF vectorizer
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],  # N-gram range for the TF-IDF vectorizer (unigrams, bigrams, trigrams)
    'kmeans__n_clusters': [3, 4, 5, 6, 7, 8, 9, 10]   # Number of clusters for K-Means clustering
}

# Create a GridSearchCV object with the pipeline and parameter grid
grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=2, n_jobs=-1)  # Using 5-fold cross-validation | verbose=2 for detailed output | n_jobs=-1 for parallel processing

# Fit the GridSearchCV object on the cleaned review text
grid_search.fit(golf_course_review['cleaned_review_text'])

# Print the best parameters and best score from the GridSearchCV
print("Best parameters:", grid_search.best_params_)
print("Best clustering score:", grid_search.best_score_)

# Use the best estimator from the GridSearchCV to transform the cleaned review text into TF-IDF vectors
best_model = grid_search.best_estimator_
tfidf_matrix = best_model.named_steps['tfidf'].transform(golf_course_review['cleaned_review_text'])

# Conver the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=best_model.named_steps['tfidf'].get_feature_names_out(), index=golf_course_review['review_id'])

# Calculate the silhouette score for the best model
labels = best_model.named_steps['kmeans'].labels_
silhouette = silhouette_score(tfidf_matrix, labels)
print("Silhouette Score:", silhouette)
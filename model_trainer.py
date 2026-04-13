import pandas as pd
import numpy as np
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import os


DATA_FILE = 'fake_job_postings.csv' 
MODEL_PATH = 'fake_job_detector_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
STEMMER = PorterStemmer()
STOPWORDS_SET = set(stopwords.words('english'))

# --- 1. DATA PREPROCESSING FUNCTIONS ---

def clean_text(text):
    """Performs stemming and stops words removal."""
    if isinstance(text, str):
        # Remove HTML tags (common in job descriptions)
        text = re.sub('<.*?>', ' ', text)
        # Remove non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = text.lower()
        text = text.split()
        # Apply stemming and remove stopwords
        text = [STEMMER.stem(word) for word in text if word not in STOPWORDS_SET]
        return ' '.join(text)
    return ''

def combine_features(row):
    """Concatenates relevant text columns for training."""
    # Combining title, description, company_profile, and requirements for a rich feature set
    features = [
        row['title'], 
        row['company_profile'], 
        row['description'], 
        row['requirements']
    ]
    # Replace NaN with empty string for concatenation
    safe_features = [str(f) if pd.notna(f) else '' for f in features]
    return ' '.join(safe_features)

def train_and_save_model():
    """Loads data, trains the model, and saves the vectorizer and model."""
    try:
        print(f"Loading data from {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please download the dataset from Kaggle and place it in the same directory.")
        return

    # Drop rows with missing target variable
    df.dropna(subset=['fraudulent'], inplace=True)
    
    # Apply feature engineering (combination)
    df['combined_text'] = df.apply(combine_features, axis=1)

    # Apply text cleaning
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    X = df['cleaned_text']
    y = df['fraudulent']

    # Splitting the data
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- 2. FEATURE EXTRACTION (TF-IDF) ---
    print("Fitting TF-IDF Vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # --- 3. MODEL TRAINING ---
    print("Training Multinomial Naive Bayes Model...")
    model = MultinomialNB(alpha=0.1) # Alpha is smoothing parameter
    model.fit(X_train_tfidf, y_train)

    # --- 4. EVALUATION ---
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n--- Model Evaluation ---")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # --- 5. SAVING MODEL AND VECTORIZER ---
    print(f"\nSaving model to {MODEL_PATH} and vectorizer to {VECTORIZER_PATH}...")
    with open(MODEL_PATH, 'wb') as model_file:
        pickle.dump(model, model_file)
    
    with open(VECTORIZER_PATH, 'wb') as vec_file:
        pickle.dump(tfidf_vectorizer, vec_file)
        
    print("Training Complete. Model and Vectorizer saved successfully.")


if __name__ == '__main__':
    print("Starting Model Training Process...")
    # NOTE: You may need to manually download the fake_job_postings.csv file first.
    train_and_save_model()
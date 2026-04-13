import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
import nltk
nltk.download('stopwords')

import streamlit as st

st.title("Detect Page Working ✅")
st.write("If you see this, navigation is correct")


    # your ML code here

# --- MODEL LOADING & PREPROCESSING ---
MODEL_PATH = 'fake_job_detector_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'
STEMMER = PorterStemmer()
STOPWORDS_SET = set(stopwords.words('english'))

# Preprocessing function (must match the one used in model_trainer.py)
def clean_text_for_prediction(text):
    """Performs stemming and stops words removal for a single input."""
    if isinstance(text, str):
        text = re.sub('<.*?>', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [STEMMER.stem(word) for word in text if word not in STOPWORDS_SET]
        return ' '.join(text)
    return ''

@st.cache_resource
def load_resources():
    """Loads the pre-trained model and vectorizer."""
    try:
        with open(MODEL_PATH, 'rb') as model_file:
            model = pickle.load(model_file)
        with open(VECTORIZER_PATH, 'rb') as vec_file:
            vectorizer = pickle.load(vec_file)
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"Error: Model or Vectorizer file not found. Please run 'python model_trainer.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()

model, vectorizer = load_resources()

# --- STREAMLIT UI ---

st.set_page_config(page_title="Fake Job Detector", layout="wide")

st.title("🤖 Automated Fake Job Detector")
st.markdown("### Using NLP and Machine Learning to Combat Recruitment Fraud")

st.info("Paste the full job description (including title, company details, requirements, etc.) into the text area below to check its authenticity.")

# Text Area for Job Description Input
job_text = st.text_area(
    "Job Advertisement Content",
    height=300,
    placeholder="Paste the Job Title, Company Profile, Description, and Requirements here...",
)

# Prediction Button
if st.button("Analyze Job Post", type="primary"):
    if job_text:
        with st.spinner('Analyzing text features...'):
            # 1. Preprocess the input text
            cleaned_input = clean_text_for_prediction(job_text)
            
            # 2. Vectorize the cleaned text
            input_vector = vectorizer.transform([cleaned_input])
            
            # 3. Predict the label
            prediction = model.predict(input_vector)[0]
            
            # 4. Get probability scores
            probabilities = model.predict_proba(input_vector)
            fraud_proba = probabilities[0][1] # Probability of being fraudulent (assuming 1 is fraudulent)

            st.subheader("Prediction Result:")
            
            if prediction == 0:
                st.success(f"✅ REAL JOB POSTING")
                st.markdown(f"The model has high confidence ({(1 - fraud_proba) * 100:.2f}%) that this is a **Legitimate Job**.")
                st.balloons()
            else:
                st.error(f"🚨 FAKE/FRAUDULENT JOB ALERT")
                st.markdown(f"The model has high confidence ({fraud_proba * 100:.2f}%) that this is a **Recruitment Scam**. Proceed with caution.")
                st.warning("Typical signs of fraud include requests for payment, promises of immediate high salary, or vague requirements.")
    else:
        st.warning("Please paste some job content to analyze.")

st.markdown(
    """
    ---
    This application utilizes a Multinomial Naive Bayes classifier trained on various textual features of job advertisements.
    """
)
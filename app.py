import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the model, TF-IDF vectorizer, and OneHotEncoder
@st.cache_resource
def load_models():
    with open('fraud_detection_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('one_hot_encoder.pkl', 'rb') as f:
        ohe = pickle.load(f)
    return model, tfidf, ohe

model, tfidf, ohe = load_models()

# Setup NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[\d]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def get_median_salary(salary_range):
    try:
        low, high = map(int, salary_range.split('-'))
        return (low + high) / 2
    except:
        return None

# Streamlit app
st.title("Fraudulent Job Posting Detection")
st.write("Enter the job posting details below to predict if it's legitimate or fraudulent.")

# Input form
with st.form(key='job_form'):
    title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
    location = st.text_input("Location", placeholder="e.g., New York, NY")
    description = st.text_area("Job Description", placeholder="Enter job description")
    requirements = st.text_area("Requirements", placeholder="Enter job requirements")
    company_profile = st.text_area("Company Profile", placeholder="Enter company profile")
    employment_type = st.selectbox("Employment Type", ["Unknown", "Full-time", "Part-time", "Contract", "Temporary", "Other"])
    industry = st.text_input("Industry", placeholder="e.g., Information Technology")
    department = st.text_input("Department", placeholder="e.g., Engineering")
    salary = st.text_input("Salary Range", placeholder="e.g., 50000-70000")
    submit_button = st.form_submit_button(label="Predict")

# Prediction logic
if submit_button:
    # Create DataFrame from inputs
    data = {
        'title': [title or ''],
        'location': [location or ''],
        'description': [description or ''],
        'requirements': [requirements or ''],
        'company_profile': [company_profile or ''],
        'employment_type': [employment_type],
        'industry': [industry or 'Unknown'],
        'department': [department or 'Unknown'],
        'salary': [salary or '']
    }
    df = pd.DataFrame(data)

    # Process salary
    df['median_salary'] = df['salary'].apply(get_median_salary)
    if df['median_salary'].notna().any():
        df['normalized_salary'] = (df['median_salary'] - df['median_salary'].min()) / (df['median_salary'].max() - df['median_salary'].min())
    else:
        df['normalized_salary'] = 0
    df = df.drop(['salary', 'median_salary'], axis=1)

    # Combine and clean text
    text_features = ['title', 'location', 'description', 'requirements', 'company_profile']
    df['combined_text'] = df[text_features].apply(lambda row: ' '.join(row.fillna('')), axis=1)
    df['combined_text'] = df['combined_text'].apply(clean_text)

    # Vectorize text
    X_text = tfidf.transform(df['combined_text'])

    # Encode categorical features
    cat_features = ['employment_type', 'industry', 'department']
    df[cat_features] = df[cat_features].fillna('Unknown')
    X_cat = ohe.transform(df[cat_features])

    # Numerical features
    numeric_features = ['normalized_salary']
    df[numeric_features] = df[numeric_features].fillna(0)
    X_num = csr_matrix(df[numeric_features].values)

    # Combine features
    X_final = hstack([X_text, X_cat, X_num])

    # Make prediction
    prediction = model.predict(X_final)[0]
    result = 'Fraudulent' if prediction == 1 else 'Legitimate'

    # Display result with color
    color = 'red' if result == 'Fraudulent' else 'green'
    st.markdown(f"<h3 style='color: {color};'>The job posting is predicted to be: <strong>{result}</strong></h3>", unsafe_allow_html=True)

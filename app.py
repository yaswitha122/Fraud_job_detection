import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load saved model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    model = pickle.load(open('fraud_detection_model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    ohe = pickle.load(open('one_hot_encoder.pkl', 'rb'))
    return model, tfidf, ohe

model, tfidf, ohe = load_model_and_preprocessors()

# Define text cleaning function
def clean_text(text):
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'[\d]+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Define salary processing function
def get_median_salary(salary):
    try:
        low, high = map(int, salary.split('-'))
        return (low + high) / 2
    except:
        return 0

def normalize_salary(median_salary):
    # Use fixed min/max from training data to avoid recalculating
    min_salary = 0  # Adjust based on your training data if needed
    max_salary = 1000000  # Adjust based on your training data if needed
    if median_salary == 0:
        return 0
    return (median_salary - min_salary) / (max_salary - min_salary)

# Streamlit app
st.title("Fraudulent Job Post Detection")
st.write("Enter job posting details to check if it's fraudulent or legitimate.")

# Input form
with st.form(key='job_form'):
    st.subheader("Job Details")
    title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
    location = st.text_input("Location", placeholder="e.g., New York, NY")
    description = st.text_area("Job Description", placeholder="e.g., We are looking for a skilled engineer...")
    requirements = st.text_area("Requirements", placeholder="e.g., 5+ years of experience...")
    company_profile = st.text_area("Company Profile", placeholder="e.g., We are a leading tech company...")

    st.subheader("Additional Details")
    employment_type = st.selectbox("Employment Type", options=['Unknown', 'Full-time', 'Part-time', 'Contract', 'Temporary', 'Other'])
    industry = st.text_input("Industry", placeholder="e.g., Information Technology")
    department = st.text_input("Department", placeholder="e.g., Engineering")
    salary = st.text_input("Salary Range (e.g., 50000-70000)", placeholder="e.g., 50000-70000")

    submit_button = st.form_submit_button(label="Predict")

# Process input and make prediction
if submit_button:
    if not all([title, location, description, requirements, company_profile, employment_type, industry, department]):
        st.error("Please fill in all required fields.")
    else:
        # Create DataFrame for input
        input_data = pd.DataFrame({
            'title': [title],
            'location': [location],
            'description': [description],
            'requirements': [requirements],
            'company_profile': [company_profile],
            'employment_type': [employment_type],
            'industry': [industry],
            'department': [department],
            'salary': [salary]
        })

        # Process salary
        median_salary = get_median_salary(salary)
        normalized_salary = normalize_salary(median_salary)
        input_data['normalized_salary'] = normalized_salary
        input_data = input_data.drop('salary', axis=1)

        # Clean text features
        text_features = ['title', 'location', 'description', 'requirements', 'company_profile']
        for col in text_features:
            input_data[col] = input_data[col].apply(clean_text)

        # Transform text features with TF-IDF
        text_vectors = [tfidf.transform(input_data[col]) for col in text_features]
        X_text = hstack(text_vectors)

        # Encode categorical features
        cat_features = ['employment_type', 'industry', 'department']
        X_cat = ohe.transform(input_data[cat_features])

        # Prepare numerical feature
        X_num = csr_matrix(input_data[['normalized_salary']].values)

        # Combine features
        X_final = hstack([X_text, X_cat, X_num])

        # Make prediction
        prediction = model.predict(X_final)[0]
        probability = model.predict_proba(X_final)[0] if hasattr(model, 'predict_proba') else None

        # Display result
        st.subheader("Prediction Result")
        if prediction == 0:
            st.success("This job posting is predicted to be **Legitimate**.")
        else:
            st.error("This job posting is predicted to be **Fraudulent**.")
        
        if probability is not None:
            st.write(f"Confidence: {probability[prediction]*100:.2f}%")

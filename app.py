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

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the saved model, TF-IDF vectorizer, and OneHotEncoder
try:
    model = pickle.load(open('fraud_detection_model.pkl', 'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    ohe = pickle.load(open('one_hot_encoder.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or encoder files not found. Please ensure 'fraud_detection_model.pkl', 'tfidf_vectorizer.pkl', and 'one_hot_encoder.pkl' are in the app directory.")
    st.stop()

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define text cleaning function
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove digits and punctuation
    text = re.sub(r'[\d]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove newlines and extra spaces
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords and lemmatize
    words = text.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Define function to calculate median salary
def get_median_salary(salary):
    try:
        low, high = map(int, salary.split('-'))
        return (low + high) / 2
    except:
        return 0

# Streamlit app
st.title("Fraudulent Job Post Detection")
st.markdown("""
This application predicts whether a job posting is fraudulent based on its details.
Enter the job posting information below and click **Predict** to see the result.
""")

# Input form
with st.form("job_post_form"):
    title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
    location = st.text_input("Location", placeholder="e.g., New York, NY")
    description = st.text_area("Job Description", placeholder="Enter the job description")
    requirements = st.text_area("Requirements", placeholder="Enter the job requirements")
    company_profile = st.text_area("Company Profile", placeholder="Enter the company profile")
    employment_type = st.selectbox("Employment Type", options=["Full-time", "Part-time", "Contract", "Temporary", "Unknown"])
    industry = st.text_input("Industry", placeholder="e.g., Information Technology")
    department = st.text_input("Department", placeholder="e.g., Engineering")
    salary = st.text_input("Salary Range (e.g., 50000-70000)", placeholder="e.g., 50000-70000")

    submitted = st.form_submit_button("Predict")

# Process input and make prediction
if submitted:
    if not (title or location or description or requirements or company_profile):
        st.warning("Please provide at least one text field (e.g., Job Title, Description).")
    else:
        # Prepare input data
        input_data = {
            'title': title if title else '',
            'location': location if location else '',
            'description': description if description else '',
            'requirements': requirements if requirements else '',
            'company_profile': company_profile if company_profile else '',
            'employment_type': employment_type,
            'industry': industry if industry else 'Unknown',
            'department': department if department else 'Unknown',
            'salary': salary
        }

        # Create DataFrame
        df_input = pd.DataFrame([input_data])

        # Process salary
        df_input['median_salary'] = df_input['salary'].apply(get_median_salary)
        min_salary = df_input['median_salary'].min()
        max_salary = df_input['median_salary'].max()
        if max_salary > min_salary:
            df_input['normalized_salary'] = (df_input['median_salary'] - min_salary) / (max_salary - min_salary)
        else:
            df_input['normalized_salary'] = 0
        df_input = df_input.drop(['salary', 'median_salary'], axis=1)

        # Clean text features
        text_features = ['title', 'location', 'description', 'requirements', 'company_profile']
        for col in text_features:
            df_input[col] = df_input[col].astype(str).apply(clean_text)

        # Combine text features
        df_input['combined_text'] = (
            df_input['title'] + ' ' +
            df_input['location'] + ' ' +
            df_input['description'] + ' ' +
            df_input['requirements'] + ' ' +
            df_input['company_profile']
        )
        df_input['combined_text'] = df_input['combined_text'].apply(clean_text)

        # Vectorize text
        X_text = tfidf.transform(df_input['combined_text'])

        # Encode categorical features
        cat_features = ['employment_type', 'industry', 'department']
        X_cat = ohe.transform(df_input[cat_features])

        # Numeric features
        X_num = csr_matrix(df_input[['normalized_salary']].values)

        # Combine features
        X_final = hstack([X_text, X_cat, X_num])

        # Make prediction
        prediction = model.predict(X_final)[0]

        # Display result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️ This job posting is predicted to be **Fraudulent**.")
        else:
            st.success("✅ This job posting is predicted to be **Legitimate**.")

# Footer
st.markdown("---")
st.markdown("Developed by Team: N. VSNSPS PRANAVI, B. ROHINI SANKARI, J. YASWITHA, M. LILLY, B. GOWTHAMI BAI")
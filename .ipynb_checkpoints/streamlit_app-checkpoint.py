import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.markdown("Fill the customer details below to predict if they are likely to churn.")

# Define input fields
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2000.0)

# Create a dictionary from inputs
input_dict = {
    'gender': 1 if gender == 'Male' else 0,
    'SeniorCitizen': SeniorCitizen,
    'Partner': 1 if Partner == 'Yes' else 0,
    'Dependents': 1 if Dependents == 'Yes' else 0,
    'tenure': tenure,
    'PhoneService': 1 if PhoneService == 'Yes' else 0,
    'PaperlessBilling': 1 if PaperlessBilling == 'Yes' else 0,
    'MonthlyCharges': MonthlyCharges,
    'TotalCharges': TotalCharges,

    # One-hot encodings
    'MultipleLines_No': 1 if MultipleLines == 'No' else 0,
    'MultipleLines_Yes': 1 if MultipleLines == 'Yes' else 0,

    'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
    'InternetService_No': 1 if InternetService == 'No' else 0,

    'OnlineSecurity_Yes': 1 if OnlineSecurity == 'Yes' else 0,
    'OnlineSecurity_No internet service': 1 if OnlineSecurity == 'No internet service' else 0,

    'OnlineBackup_Yes': 1 if OnlineBackup == 'Yes' else 0,
    'OnlineBackup_No internet service': 1 if OnlineBackup == 'No internet service' else 0,

    'DeviceProtection_Yes': 1 if DeviceProtection == 'Yes' else 0,
    'DeviceProtection_No internet service': 1 if DeviceProtection == 'No internet service' else 0,

    'TechSupport_Yes': 1 if TechSupport == 'Yes' else 0,
    'TechSupport_No internet service': 1 if TechSupport == 'No internet service' else 0,

    'StreamingTV_Yes': 1 if StreamingTV == 'Yes' else 0,
    'StreamingTV_No internet service': 1 if StreamingTV == 'No internet service' else 0,

    'StreamingMovies_Yes': 1 if StreamingMovies == 'Yes' else 0,
    'StreamingMovies_No internet service': 1 if StreamingMovies == 'No internet service' else 0,

    'Contract_One year': 1 if Contract == 'One year' else 0,
    'Contract_Two year': 1 if Contract == 'Two year' else 0,

    'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
    'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
    'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0
}

# Create input DataFrame
input_df = pd.DataFrame([input_dict])

# Reorder columns to match training data
required_features = model.feature_names_in_  # Only available in newer sklearn
input_df = input_df.reindex(columns=required_features, fill_value=0)

# Scale numerical features
input_scaled = scaler.transform(input_df)

# Predict
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn. (Confidence: {probability:.2%})")
    else:
        st.success(f"✅ This customer is likely to stay. (Confidence: {1 - probability:.2%})")


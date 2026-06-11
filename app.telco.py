import streamlit as st
import joblib
import pandas as pd

# --- Configuration and Model Loading (done once when the app starts) ---
# Ensure these files are in the same directory as your streamlit_app.py
# or provide the correct path if they are in a subfolder.
model = joblib.load('xgboost_model.joblib')
scaler = joblib.load('scaler.joblib')

TRAINING_COLUMNS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

# --- Streamlit UI and Prediction Logic ---
st.title('Telco Customer Churn Prediction')

st.write('Enter customer details to predict churn:')

# Example: Create input widgets for features
tenure = st.slider('Tenure (months)', 0, 72, 36)
monthly_charges = st.number_input('Monthly Charges', 0.0, 200.0, 50.0)
total_charges = st.number_input('Total Charges', 0.0, 10000.0, 1000.0)
senior_citizen = st.checkbox('Senior Citizen')
gender = st.radio('Gender', ['Male', 'Female'])
# ... add more input widgets for all your features

# Collect user input into a dictionary
user_input = {
    'SeniorCitizen': 1 if senior_citizen else 0,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'gender_Male': True if gender == 'Male' else False,
    # ... populate all other features based on your TRAINING_COLUMNS
}

# Create a DataFrame from user input
input_df = pd.DataFrame([user_input])

# Preprocessing steps (must match training preprocessing)
input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
input_df.fillna(0, inplace=True) # Or a more robust imputation

# One-hot encode new categorical inputs (if any, in this simplified example, they're already boolean)
# For a full app, you'd apply get_dummies to the raw categorical columns, then reindex
# input_df = pd.get_dummies(input_df, drop_first=True) # Only if you have raw categoricals in user_input

# Reindex to ensure all columns (and their order) match the training data
input_df = input_df.reindex(columns=TRAINING_COLUMNS, fill_value=0)

# Scale numerical features
scaled_features = scaler.transform(input_df)

if st.button('Predict Churn'):
    prediction = model.predict(scaled_features)
    prediction_proba = model.predict_proba(scaled_features)[:, 1][0]

    if prediction[0] == 1:
        st.error(f'Prediction: Customer is likely to CHURN (Probability: {prediction_proba:.2f})')
    else:
        st.success(f'Prediction: Customer is NOT likely to churn (Probability: {prediction_proba:.2f})')

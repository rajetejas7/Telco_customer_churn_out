import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('xgboost_model.joblib')
scaler = joblib.load('scaler.joblib')

# Define the expected columns after one-hot encoding, as derived from the training data
# This list should match the `final_columns_after_dummies` list generated above
# NOTE: Replace 'final_columns_after_dummies' with the actual list of strings if you are running this outside of the Colab notebook where that variable is defined.
# For example: TRAINING_COLUMNS = ['SeniorCitizen', 'tenure', ...]
# Based on our previous steps, the variable `final_columns_after_dummies` holds the correct list.
TRAINING_COLUMNS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes', 'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        # Convert input to DataFrame, ensuring it has all original columns
        query_df = pd.DataFrame(json_, index=[0])

        # --- Preprocessing steps (must match training preprocessing) ---
        # Convert TotalCharges to numeric, handling errors
        query_df['TotalCharges'] = pd.to_numeric(query_df['TotalCharges'], errors='coerce')
        # Handle potential NaNs, for prediction it's better to impute or ensure input quality
        # For simplicity, we'll fill with 0 here, but a more robust solution might use the mean/median from training
        query_df.fillna(0, inplace=True) # This is a simplification; consider a more robust imputation if NaNs are expected

        # Apply one-hot encoding
        query_df = pd.get_dummies(query_df, drop_first=True)

        # Reindex to ensure all columns (and their order) match the training data
        # Fill missing columns with 0 and drop extra columns
        query_df = query_df.reindex(columns=TRAINING_COLUMNS, fill_value=0)

        # Scale numerical features
        scaled_features = scaler.transform(query_df)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        
        # Get prediction probabilities if the model supports it
        try:
            prediction_proba = model.predict_proba(scaled_features)[:, 1][0] # Probability of 'Yes' class
            return jsonify({'churn_prediction': int(prediction[0]), 'churn_probability': float(prediction_proba)})
        except AttributeError:
            return jsonify({'churn_prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Use host='0.0.0.0' to make the app accessible externally in a Colab environment
    app.run(host='0.0.0.0', port=5000)

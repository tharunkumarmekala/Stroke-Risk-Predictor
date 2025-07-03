from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import xgboost as xgb

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Define model paths relative to the current script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    'xgboost_json': os.path.join(BASE_DIR, 'models', 'strokemodel.json'),
    'xgboost_pkl': os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl'),
    'ensemble': os.path.join(BASE_DIR, 'models', 'ensemble_model.pkl'),
    'scaler': os.path.join(BASE_DIR, 'models', 'scaler.pkl') # Assuming you might still use a scaler
}

# Global variables for model and scaler
model = None
model_type = None
scaler = None

# Columns in the exact order your model expects AFTER mapping/encoding
# THIS IS CRITICAL. MUST MATCH YOUR MODEL'S TRAINING DATA.
MODEL_FEATURE_COLUMNS = [
    'Age', 'Sex', 'BMI', 'Cholesterol', 'Hypertension',
    'Atrial_Fibrillation', 'Diabetes', 'Smoking', 'Previous_Stroke'
]

# Load models and scaler on application startup
@app.before_first_request
def load_resources():
    global model, model_type, scaler
    
    # Try loading XGBoost from JSON first
    try:
        temp_model = xgb.Booster()
        temp_model.load_model(MODEL_PATHS['xgboost_json'])
        model = temp_model
        model_type = 'xgboost_json'
        print(f"Loaded XGBoost model from JSON: {MODEL_PATHS['xgboost_json']}")
    except Exception as e_json:
        print(f"Could not load XGBoost from JSON: {e_json}")
        try:
            # Fall back to pickle version
            temp_model = joblib.load(MODEL_PATHS['xgboost_pkl'])
            model = temp_model
            model_type = 'xgboost_pkl'
            print(f"Loaded XGBoost model from pickle: {MODEL_PATHS['xg xgboost_model.pkl']}")
        except Exception as e_pkl:
            print(f"Could not load XGBoost from pickle: {e_pkl}")
            try:
                # Final fallback to ensemble
                temp_model = joblib.load(MODEL_PATHS['ensemble'])
                model = temp_model
                model_type = 'ensemble'
                print(f"Loaded ensemble model: {MODEL_PATHS['ensemble']}")
            except Exception as e_ensemble:
                print(f"Could not load any model: {e_ensemble}")
                model = None # If no model loads, set model to None

    # Load scaler if exists
    try:
        scaler = joblib.load(MODEL_PATHS['scaler'])
        print(f"Loaded scaler from: {MODEL_PATHS['scaler']}")
    except FileNotFoundError:
        print(f"No scaler found at {MODEL_PATHS['scaler']}")
        scaler = None
    except Exception as e_scaler:
        print(f"Error loading scaler: {e_scaler}")
        scaler = None

    if model is None:
        print("CRITICAL ERROR: No prediction model could be loaded.")
    
# Helper functions to map form values to model expected values
# These mappings MUST exactly match how your model was trained
def _map_age(age_value):
    # 'age' from HTML is 30, 50, 65. Directly use these as numerical values.
    return float(age_value)

def _map_bmi(bmi_value):
    # 'bmi' from HTML is 18, 22, 27, 32. Directly use these as numerical values.
    return float(bmi_value)

def _map_cholesterol(chol_value):
    # 'cholesterol' from HTML is 170, 220, 250. Directly use these as numerical values.
    return float(chol_value)

# Main route to serve the HTML file
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    global model, model_type, scaler # Access global variables

    if model is None:
        return jsonify({
            'error': 'Model Not Loaded',
            'message': 'The prediction model could not be loaded on the server. Please check server logs.'
        }), 500

    try:
        # Get data from POST request (now using JSON as per the HTML's fetch request)
        data = request.get_json()
        print("Received JSON data:", data)

        # Map the form data to match your model's expected input format
        # Ensure 'sex' mapping (male=1, female=0) matches your training
        input_data_mapped = {
            'Age': _map_age(data['age']),
            'Sex': 1 if data['sex'] == 'male' else 0,
            'BMI': _map_bmi(data['bmi']),
            'Cholesterol': _map_cholesterol(data['cholesterol']),
            'Hypertension': float(data['hypertension']), # Ensure float for scaling consistency
            'Atrial_Fibrillation': float(data['atrial_fibrillation']), # Ensure float
            'Diabetes': float(data['diabetes']), # Ensure float
            'Smoking': float(data['smoking']), # Ensure float
            'Previous_Stroke': float(data['previous_stroke']) # Ensure float
        }
        
        # Create DataFrame ensuring correct column order
        df = pd.DataFrame([input_data_mapped], columns=MODEL_FEATURE_COLUMNS)
        print("DataFrame before scaling:\n", df)
        
        # Scale features if scaler exists
        if scaler is not None:
            df_scaled_array = scaler.transform(df)
            df = pd.DataFrame(df_scaled_array, columns=df.columns) # Convert back to DataFrame
            print("DataFrame after scaling:\n", df)
        
        # Make prediction
        prediction_proba = 0.0 # Default if no model or issue
        if model_type == 'xgboost_json':
            dmatrix = xgb.DMatrix(df)
            prediction_proba = model.predict(dmatrix)[0] # XGBoost Booster.predict returns probabilities directly
        else: # For sklearn-style models (like xgboost_model.pkl or ensemble_model.pkl if they're sklearn-API)
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(df)[0][1] # Probability of stroke (class 1)
            else:
                # Fallback if model only has predict (returns class, not prob) - less ideal for risk assessment
                prediction_proba = model.predict(df)[0]
                prediction_proba = float(prediction_proba) # Convert class (0 or 1) to float

        # Convert to percentage (0-100%) with clipping
        risk_percentage = min(100.0, max(0.0, float(prediction_proba) * 100))
        
        # Return only the risk percentage, as all other detailed logic is in frontend JS
        return jsonify({
            'risk_percentage': risk_percentage,
            'status': 'success'
        })
        
    except (ValueError, TypeError, KeyError) as e:
        print(f"Input validation error: {e}")
        return jsonify({
            'error': 'Invalid Input Data',
            'message': f"There was an issue processing your input. Please check all fields. Details: {str(e)}"
        }), 400
    except Exception as e:
        print(f"Unexpected error during prediction: {e}")
        return jsonify({
            'error': 'Prediction Failed',
            'message': f"An unexpected server error occurred: {str(e)}. Please try again later."
        }), 500

# This block is for local development only
if __name__ == '__main__':
    # Ensure 'models' directory exists for local testing
    if not os.path.exists(os.path.join(BASE_DIR, 'models')):
        print(f"Creating 'models' directory at {os.path.join(BASE_DIR, 'models')}")
        os.makedirs(os.path.join(BASE_DIR, 'models'))
    
    # Load resources for local run
    load_resources() 
    
    # Check if a model was loaded
    if model is None:
        print("Application cannot start locally because no model was loaded. Please ensure model files are present.")
        exit(1) # Exit with an error code

    app.run(debug=True)

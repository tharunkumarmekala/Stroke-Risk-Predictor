import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS # New import for CORS
import numpy as np
import shap # Make sure shap is installed in your Vercel deployment via requirements.txt
import os # New import for path manipulation

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Robustly load your pre-trained model ---
# Use os.path.dirname(__file__) to get the directory of the current script (app.py)
# and then join it with the model filename. This makes the path relative to the script.
model_path = os.path.join(os.path.dirname(__file__), 'xgb_tuned_model_optimized_f1.pkl')

model = None
explainer = None # Initialize explainer outside try-except
try:
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model) # Initialize the SHAP TreeExplainer with the loaded model
    print("Model and SHAP explainer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model: {e}. Ensure '{model_path}' exists.")
    # Critical error, the app won't function without the model
except Exception as e:
    print(f"An unexpected error occurred while loading the model or explainer: {e}")
    # Critical error, the app won't function without the model

# --- HARDCODED PREPROCESSING PARAMETERS ---
# !! IMPORTANT !! REPLACE THESE PLACEHOLDER VALUES WITH THE EXACT ONES FROM YOUR TRAINING SCRIPT OUTPUT
# If these don't match, your predictions and SHAP values will be incorrect.

LABEL_ENCODER_MAPPINGS = {
    'gender': {'Female': 0, 'Male': 1}, # Example: CHECK YOUR TRAINING SCRIPT'S OUTPUT!
    'ever_married': {'No': 0, 'Yes': 1}, # Example: CHECK YOUR TRAINING SCRIPT'S OUTPUT!
    'work_type': {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self-employed': 3, 'children': 4}, # Example: CHECK YOUR TRAINING SCRIPT'S OUTPUT!
    'Residence_type': {'Rural': 0, 'Urban': 1}, # Example: CHECK YOUR TRAINING SCRIPT'S OUTPUT!
    'smoking_status': {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3} # Example: CHECK YOUR TRAINING SCRIPT'S OUTPUT!
}

# Inverse mappings for display purposes (converting encoded numbers back to original strings)
INVERSE_LABEL_ENCODER_MAPPINGS = {}
for col, mapping in LABEL_ENCODER_MAPPINGS.items():
    INVERSE_LABEL_ENCODER_MAPPINGS[col] = {v: k for k, v in mapping.items()}


SCALER_MEAN = np.array([ # REPLACE THIS WITH YOUR SCALER.MEAN_ OUTPUT!
    0.509749,   # gender (after label encoding)
    43.238914,  # age
    0.097507,   # hypertension
    0.041077,   # heart_disease
    0.655812,   # ever_married (after label encoding)
    1.424364,   # work_type (after label encoding)
    0.508535,   # Residence_type (after label encoding)
    106.136053, # avg_glucose_level
    28.894562,  # bmi (after imputation, this is the mean of imputed column)
    1.370395    # smoking_status (after label encoding)
])

SCALER_SCALE = np.array([ # REPLACE THIS WITH YOUR SCALER.SCALE_ OUTPUT (standard deviation)!
    0.499951,   # gender
    22.646549,  # age
    0.296653,   # hypertension
    0.198466,   # heart_disease
    0.475253,   # ever_married
    0.985920,   # work_type
    0.499951,   # Residence_type
    45.244358,  # avg_glucose_level
    7.697669,   # bmi
    1.026365    # smoking_status
])

# Define the exact order of features your model expects *before* scaling
EXPECTED_FEATURES_ORDER_BEFORE_SCALING = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

# --- Risk Level Thresholds (Adjust as needed based on your model's performance) ---
RISK_THRESHOLDS = {
    'low': 0.10,
    'moderate': 0.30
}

@app.route('/')
def home():
    # Vercel needs to know to serve index.html from the templates folder.
    # The default Flask setup for templates usually works without path adjustments here.
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None: # Check if model loaded successfully during app startup
        return jsonify({
            'error': 'Server Configuration Error',
            'message': 'Prediction service is not fully initialized. Model file could not be loaded on startup.'
        }), 500

    try:
        data = request.form.to_dict()
        required_keys = [
            'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
        ]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing form data for: {key}")

        print("Received form data:", data)

        # --- Preprocessing: Convert form data to model-compatible format ---
        processed_input_dict = {}

        # Numerical features
        processed_input_dict['age'] = float(data['age'])
        processed_input_dict['avg_glucose_level'] = float(data['avg_glucose_level'])
        processed_input_dict['bmi'] = float(data['bmi'])

        # Binary features (Yes/No to 1/0)
        processed_input_dict['hypertension'] = 1 if data['hypertension'] == 'Yes' else 0
        processed_input_dict['heart_disease'] = 1 if data['heart_disease'] == 'Yes' else 0

        # LabelEncoded categorical features
        categorical_cols_to_encode = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_cols_to_encode:
            mapping = LABEL_ENCODER_MAPPINGS.get(col)
            if mapping is None:
                return jsonify({'error': f"LabelEncoder mapping not found for {col}.", 'message': "Server preprocessing configuration error."}), 500

            value = data[col]
            if value not in mapping:
                 return jsonify({
                    'error': f"Invalid input for {col}.",
                    'message': f"The selected value '{value}' for {col} is not recognized by the model. "
                               "Please select a valid option from the form."
                }), 400
            processed_input_dict[col] = mapping[value]

        # Create a Pandas DataFrame with the exact feature order
        input_df_before_scaling = pd.DataFrame([processed_input_dict], columns=EXPECTED_FEATURES_ORDER_BEFORE_SCALING)

        print("Input DataFrame before scaling:\n", input_df_before_scaling)

        # Apply StandardScaler transformation using hardcoded mean and scale
        input_scaled_array = (input_df_before_scaling.values - SCALER_MEAN) / SCALER_SCALE

        # Create a DataFrame for SHAP, ensuring column names are preserved after scaling
        input_for_shap = pd.DataFrame(input_scaled_array, columns=EXPECTED_FEATURES_ORDER_BEFORE_SCALING)

        # --- Make prediction ---
        prediction_proba = model.predict_proba(input_scaled_array)[:, 1][0] # Probability of stroke (class 1)

        # --- Determine Risk Level and Message ---
        risk_level_text = ""
        risk_emoji = ""
        risk_message = ""

        if prediction_proba <= RISK_THRESHOLDS['low']:
            risk_level_text = "Low Risk"
            risk_emoji = "🟢"
            risk_message = "Low risk detected. Keep up the great work with your healthy lifestyle!"
        elif prediction_proba <= RISK_THRESHOLDS['moderate']:
            risk_level_text = "Moderate Risk"
            risk_emoji = "🟡"
            risk_message = "Moderate risk detected. Time to take action with lifestyle changes."
        else:
            risk_level_text = "High Risk"
            risk_emoji = "🔴"
            risk_message = "High risk detected. Immediate medical consultation recommended."

        # --- Calculate SHAP values for risk contributors ---
        risk_contributors = []
        if explainer: # Only proceed if explainer was loaded successfully
            shap_values_instance = explainer.shap_values(input_for_shap)[1][0] # Index [1] for positive class, [0] for single instance
            shap_series = pd.Series(shap_values_instance, index=input_for_shap.columns)
            positive_contributors = shap_series[shap_series > 0.001].sort_values(ascending=False)

            if not positive_contributors.empty:
                for feature, shap_val in positive_contributors.head(3).items():
                    original_value = data.get(feature, 'N/A') # Use .get() for safety
                    if feature in INVERSE_LABEL_ENCODER_MAPPINGS:
                        encoded_value = processed_input_dict.get(feature)
                        original_value = INVERSE_LABEL_ENCODER_MAPPINGS[feature].get(encoded_value, original_value)

                    contributor_description = ""
                    # Specific descriptions based on feature and value
                    if feature == 'bmi':
                        bmi_val = float(data['bmi'])
                        if bmi_val < 18.5:
                            contributor_description = f"Underweight (BMI <18.5) - May indicate poor nutrition"
                        elif bmi_val >= 25.0 and bmi_val < 30.0:
                            contributor_description = f"Overweight (BMI {bmi_val:.1f}) - Increased health risks"
                        elif bmi_val >= 30.0:
                            contributor_description = f"Obese (BMI {bmi_val:.1f}) - Significantly increased health risks"
                        else:
                            contributor_description = f"BMI ({bmi_val:.1f})"
                    elif feature == 'age':
                        contributor_description = f"Age ({int(data['age'])} years) - Age is a natural risk factor"
                    elif feature == 'avg_glucose_level':
                        contributor_description = f"High Glucose Level ({float(data['avg_glucose_level']):.1f} mg/dL)" if float(data['avg_glucose_level']) > 125 else f"Glucose Level ({float(data['avg_glucose_level']):.1f} mg/dL)"
                    elif feature == 'hypertension':
                        contributor_description = f"Hypertension ({original_value}) - High blood pressure is a major risk factor"
                    elif feature == 'heart_disease':
                        contributor_description = f"Heart Disease ({original_value}) - Pre-existing heart conditions increase risk"
                    elif feature == 'smoking_status':
                        if original_value == 'smokes':
                            contributor_description = f"Smoking ({original_value}) - Significantly increases stroke risk"
                        elif original_value == 'formerly smoked':
                            contributor_description = f"Formerly Smoked ({original_value}) - Past smoking habits can contribute"
                        else:
                            contributor_description = f"Smoking Status ({original_value})"
                    else:
                        contributor_description = f"{feature.replace('_', ' ').title()}: {original_value}" # Default fallback

                    risk_contributors.append(contributor_description)
            else:
                risk_contributors.append("No prominent specific risk contributors identified by the model for this profile.")
        else:
            risk_contributors.append("Risk contributors could not be calculated (SHAP explainer not loaded).")


        # --- Clinical Actions & Lifestyle Recommendations ---
        clinical_actions = [
            "Annual checkup - Important for prevention"
        ]
        if prediction_proba > RISK_THRESHOLDS['low']:
            clinical_actions.append("Consult with a doctor about your risk factors and specific management plan.")
        if prediction_proba > RISK_THRESHOLDS['moderate']:
            clinical_actions.append("Consider immediate medical consultation for high-risk assessment.")


        lifestyle_recommendations = [
            "Exercise regularly - 30 minutes most days",
            "Balanced diet - Focus on whole foods, fruits, and vegetables",
            "Maintain healthy weight - BMI 18.5-24.9 ideal",
            "Manage stress effectively through mindfulness or hobbies",
            "Monitor blood pressure and glucose levels regularly",
            "Limit alcohol intake",
            "Quit smoking (if applicable) - seek support if needed"
        ]

        # --- Food Recommendations by Gender ---
        food_recommendations = []
        user_gender_for_food = data.get('gender', 'Unknown') # Default to Unknown if missing
        if user_gender_for_food == 'Male':
            food_recommendations = [
                "Lean chicken 🍗", "Brown rice 🍚", "Chia seeds 🌱", "Broccoli 🥦",
                "Olive oil 🫒", "Turkey 🦃", "Lentils 🥣", "Tomatoes 🍅"
            ]
        elif user_gender_for_food == 'Female':
            food_recommendations = [
                "Salmon 🍣", "Quinoa 🍚", "Spinach 🥬", "Berries 🍓",
                "Avocado 🥑", "Greek yogurt 🍦", "Almonds 🌰", "Sweet potatoes 🍠"
            ]
        else: # Covers 'Other' (which we removed from HTML) or unexpected values
            food_recommendations = ["General healthy food choices like fruits, vegetables, and lean proteins."]

        # --- Construct the detailed JSON response ---
        response_data = {
            'risk_percentage': f"{prediction_proba * 100:.1f}%",
            'risk_level': f"{risk_emoji} {risk_level_text}",
            'risk_message': risk_message,
            'risk_contributors': risk_contributors,
            'clinical_actions': clinical_actions,
            'lifestyle_recommendations': lifestyle_recommendations,
            'food_recommendations': food_recommendations,
            'gender_for_food': user_gender_for_food
        }
        return jsonify(response_data)

    except (ValueError, KeyError) as e:
        return jsonify({
            'error': 'Invalid Input',
            'message': f"One or more input values are invalid or missing. Please check your entries. ({str(e)})"
        }), 400
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return jsonify({
            'error': 'Prediction Failed',
            'message': f"An internal server error occurred: {str(e)}. Please try again later."
        }), 500

# This block is for local development only and is ignored by Vercel
if __name__ == '__main__':
    # Make sure this is only called if running locally
    # It attempts to load the model and explainer again, which is redundant if successful above,
    # but provides clearer local startup feedback.
    if model is None:
        print("\n--- WARNING: Model not loaded during app initialization. Attempting local reload for debug. ---")
        try:
            model = joblib.load(model_path)
            global explainer # Access the global explainer variable
            explainer = shap.TreeExplainer(model)
            print("Model and SHAP explainer successfully loaded for local debug.")
        except Exception as e:
            print(f"Local debug load failed: {e}")
            print("Cannot run locally without the model. Exiting.")
            exit()
    app.run(debug=True)

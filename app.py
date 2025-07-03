import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import shap # Ensure shap is installed

app = Flask(__name__)
CORS(app)

# --- Path to your pre-trained model ---
model_path = os.path.join(os.path.dirname(__file__), 'xgb_tuned_model_optimized_f1.pkl')
model = None
explainer = None
try:
    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)
    print("Model and SHAP explainer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model: {e}. Ensure '{model_path}' exists.")
except Exception as e:
    print(f"An unexpected error occurred while loading the model or explainer: {e}")

# --- HARDCODED PREPROCESSING PARAMETERS FROM YOUR ORIGINAL ML MODEL ---
# These are the parameters your ML model (xgb_tuned_model_optimized_f1.pkl) was trained with.
# They DO NOT necessarily align perfectly with the new HTML form's inputs.
# YOU MUST ENSURE THESE ARE CORRECT FROM YOUR ORIGINAL TRAINING SCRIPT.

LABEL_ENCODER_MAPPINGS = {
    'gender': {'Female': 0, 'Male': 1},
    'ever_married': {'No': 0, 'Yes': 1}, # This input is MISSING in new HTML, will be defaulted
    'work_type': {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self-employed': 3, 'children': 4}, # Missing, will be defaulted
    'Residence_type': {'Rural': 0, 'Urban': 1}, # Missing, will be defaulted
    'smoking_status': {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}
}

INVERSE_LABEL_ENCODER_MAPPINGS = {}
for col, mapping in LABEL_ENCODER_MAPPINGS.items():
    INVERSE_LABEL_ENCODER_MAPPINGS[col] = {v: k for k, v in mapping.items()}

# Scaler means and scales from your ORIGINAL ML model's training data (X.columns)
SCALER_MEAN = np.array([
    0.509749,   # gender (encoded)
    43.238914,  # age
    0.097507,   # hypertension (from ML model's data, not new form's 'hypertension')
    0.041077,   # heart_disease (Missing, will be defaulted)
    0.655812,   # ever_married (encoded, Missing)
    1.424364,   # work_type (encoded, Missing)
    0.508535,   # Residence_type (encoded, Missing)
    106.136053, # avg_glucose_level (Missing)
    28.894562,  # bmi
    1.370395    # smoking_status (encoded)
])

SCALER_SCALE = np.array([
    0.499951,
    22.646549,
    0.296653,
    0.198466,
    0.475253,
    0.985920,
    0.499951,
    45.244358,
    7.697669,
    1.026365
])

# The EXACT order of features your ML model expects *before* scaling
EXPECTED_FEATURES_ORDER_BEFORE_SCALING = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

# --- Default values for features NOT PRESENT in the new HTML form ---
# These are crucial assumptions because your ML model requires all its original features.
DEFAULT_VALUES_FOR_ML_MODEL = {
    'heart_disease': 0,        # Assuming no heart disease by default (0 for No)
    'ever_married': LABEL_ENCODER_MAPPINGS['ever_married']['Yes'], # Assuming Married as common default
    'work_type': LABEL_ENCODER_MAPPINGS['work_type']['Private'], # Assuming Private work as common default
    'Residence_type': LABEL_ENCODER_MAPPINGS['Residence_type']['Urban'], # Assuming Urban as common default
    'avg_glucose_level': 90.0  # Common healthy average glucose level
}

# --- Risk Level Thresholds (Adjust these for the final combined risk percentage) ---
# These are used for the final percentage, not the raw ML probability anymore.
RISK_THRESHOLDS = {
    'very_low': 5,
    'low': 10,
    'moderate': 20,
    'elevated': 30,
    'high': 40,
    'serious': 60
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({
            'error': 'Server Configuration Error',
            'message': 'Prediction service is not fully initialized. Model file could not be loaded on startup.'
        }), 500

    try:
        # Request data is now JSON, not form data
        data = request.json
        print("Received JSON data:", data)

        # Basic validation for new HTML form's fields
        required_new_keys = [
            'age', 'sex', 'bmi', 'cholesterol', 'hypertension',
            'atrial_fibrillation', 'diabetes', 'smoking', 'previous_stroke'
        ]
        for key in required_new_keys:
            if key not in data or data[key] is None:
                raise ValueError(f"Missing or null input for: {key}")

        # --- Map new HTML inputs to old ML model inputs ---
        ml_input_data = {}

        # 1. Gender mapping
        ml_input_data['gender'] = LABEL_ENCODER_MAPPINGS['gender'][data['sex'].title()] # 'male'->'Male', 'female'->'Female'

        # 2. Age (from categorical to continuous - use mid-point of HTML categories)
        if data['age'] == 30: # Under 40
            ml_input_data['age'] = 30.0
        elif data['age'] == 50: # 40-59
            ml_input_data['age'] = 50.0
        elif data['age'] == 65: # 60+
            ml_input_data['age'] = 65.0
        else:
            ml_input_data['age'] = float(data['age']) # Fallback if direct number

        # 3. BMI (from categorical to continuous - use mid-point of HTML categories)
        if data['bmi'] == 18: # Underweight (<18.5)
            ml_input_data['bmi'] = 17.0
        elif data['bmi'] == 22: # Normal (18.5-24.9)
            ml_input_data['bmi'] = 22.0
        elif data['bmi'] == 27: # Overweight (25-29.9)
            ml_input_data['bmi'] = 27.0
        elif data['bmi'] == 32: # Obese (>=30)
            ml_input_data['bmi'] = 35.0 # Use a representative obese BMI
        else:
            ml_input_data['bmi'] = float(data['bmi'])

        # 4. Hypertension (New HTML's Hypertension maps to ML's Hypertension)
        # Assuming new HTML's 0='Normal', 1='Elevated', 2='High' maps to ML's 0='No', 1='Yes'
        ml_input_data['hypertension'] = 1 if data['hypertension'] > 0 else 0

        # 5. Smoking (New HTML's Smoking maps to ML's Smoking Status)
        # New: 0='Never', 1='Current/Former' -> ML: 'never smoked', 'smokes' (simplified)
        if data['smoking'] == 0:
            ml_input_data['smoking_status'] = LABEL_ENCODER_MAPPINGS['smoking_status']['never smoked']
        else: # assuming 1
            ml_input_data['smoking_status'] = LABEL_ENCODER_MAPPINGS['smoking_status']['smokes']
            # Note: This is a simplification as original model had 'formerly smoked' and 'Unknown'

        # 6. Use DEFAULT_VALUES_FOR_ML_MODEL for features missing in new HTML form
        ml_input_data['heart_disease'] = DEFAULT_VALUES_FOR_ML_MODEL['heart_disease']
        ml_input_data['ever_married'] = DEFAULT_VALUES_FOR_ML_MODEL['ever_married']
        ml_input_data['work_type'] = DEFAULT_VALUES_FOR_ML_MODEL['work_type']
        ml_input_data['Residence_type'] = DEFAULT_VALUES_FOR_ML_MODEL['Residence_type']
        ml_input_data['avg_glucose_level'] = DEFAULT_VALUES_FOR_ML_MODEL['avg_glucose_level']


        # Create DataFrame in the exact order the ML model expects
        input_df_before_scaling = pd.DataFrame([ml_input_data], columns=EXPECTED_FEATURES_ORDER_BEFORE_SCALING)

        # Apply StandardScaler transformation using hardcoded mean and scale
        input_scaled_array = (input_df_before_scaling.values - SCALER_MEAN) / SCALER_SCALE

        # Make prediction using the loaded ML model
        ml_prediction_proba = model.predict_proba(input_scaled_array)[:, 1][0]
        ml_risk_percentage = ml_prediction_proba * 100

        # --- Calculate SHAP values for ML risk contributors (based on the ML model's features) ---
        ml_risk_contributors = []
        if explainer:
            # SHAP explainer requires feature names, so use a DataFrame
            input_for_shap = pd.DataFrame(input_scaled_array, columns=EXPECTED_FEATURES_ORDER_BEFORE_SCALING)
            shap_values_instance = explainer.shap_values(input_for_shap)[1][0]
            shap_series = pd.Series(shap_values_instance, index=input_for_shap.columns)
            positive_contributors = shap_series[shap_series > 0.001].sort_values(ascending=False)

            if not positive_contributors.empty:
                for feature, shap_val in positive_contributors.head(3).items():
                    original_input_value = None
                    if feature in ['age', 'avg_glucose_level', 'bmi']:
                        original_input_value = ml_input_data[feature] # Use the numerical value we mapped
                    elif feature in ['hypertension', 'heart_disease']:
                        original_input_value = "Yes" if ml_input_data[feature] == 1 else "No"
                    elif feature in INVERSE_LABEL_ENCODER_MAPPINGS:
                        original_input_value = INVERSE_LABEL_ENCODER_MAPPINGS[feature].get(ml_input_data[feature], 'N/A')

                    contributor_description = ""
                    if feature == 'bmi':
                        if original_input_value < 18.5:
                            contributor_description = f"Underweight (BMI <18.5) - May indicate poor nutrition"
                        elif original_input_value >= 25.0 and original_input_value < 30.0:
                            contributor_description = f"Overweight (BMI {original_input_value:.1f}) - Increased health risks"
                        elif original_input_value >= 30.0:
                            contributor_description = f"Obese (BMI {original_input_value:.1f}) - Significantly increased health risks"
                        else:
                            contributor_description = f"BMI ({original_input_value:.1f})"
                    elif feature == 'age':
                        contributor_description = f"Age ({int(original_input_value)} years) - Age is a natural risk factor"
                    elif feature == 'avg_glucose_level':
                        contributor_description = f"High Glucose Level ({original_input_value:.1f} mg/dL)" if original_input_value > 125 else f"Glucose Level ({original_input_value:.1f} mg/dL)"
                    elif feature == 'hypertension':
                        contributor_description = f"Hypertension ({original_input_value}) - High blood pressure is a major risk factor"
                    elif feature == 'heart_disease':
                        contributor_description = f"Heart Disease ({original_input_value}) - Pre-existing heart conditions increase risk"
                    elif feature == 'smoking_status':
                        if original_input_value == 'smokes':
                            contributor_description = f"Smoking ({original_input_value}) - Significantly increases stroke risk"
                        elif original_input_value == 'formerly smoked':
                            contributor_description = f"Formerly Smoked ({original_input_value}) - Past smoking habits can contribute"
                        else:
                            contributor_description = f"Smoking Status ({original_input_value})"
                    else:
                        contributor_description = f"{feature.replace('_', ' ').title()}: {original_input_value}"

                    ml_risk_contributors.append(contributor_description)
            else:
                ml_risk_contributors.append("No prominent specific risk contributors identified by the ML model for this profile.")
        else:
            ml_risk_contributors.append("Risk contributors from ML model could not be calculated (SHAP explainer not loaded).")

        # Return just the ML risk percentage and its contributors for blending on frontend
        return jsonify({
            'risk_percentage': ml_risk_percentage,
            'ml_risk_contributors': ml_risk_contributors
        })

    except (ValueError, KeyError) as e:
        print(f"Input processing error: {e}")
        return jsonify({'error': 'Invalid Input', 'message': f"One or more input values are invalid or missing. ({str(e)})"}), 400
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({'error': 'Prediction Failed', 'message': f"An internal server error occurred. ({str(e)})"}), 500

if __name__ == '__main__':
    # For local development, this re-attempts to load the model.
    # In Vercel, this block is typically not executed in the same way.
    if model is None:
        try:
            model = joblib.load(model_path)
            explainer = shap.TreeExplainer(model)
            print("Model and SHAP explainer successfully loaded for local debug.")
        except Exception as e:
            print(f"Local debug load failed: {e}")
            print("Cannot run locally without the model. Exiting.")
            # exit() # Don't exit here if you want to test API error handling
    app.run(debug=True)

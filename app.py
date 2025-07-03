import joblib # Using joblib because your training code uses it
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- Load your pre-trained model and preprocessors ---
try:
    # Load the specific model saved from your training script
    model = joblib.load('xgb_tuned_model_optimized_f1.pkl')
    # Load the fitted StandardScaler
    scaler = joblib.load('scaler.pkl')
    # Load the dictionary of fitted LabelEncoders
    label_encoders = joblib.load('label_encoders.pkl')
    print("Model and preprocessors loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading required files: {e}. "
          "Ensure 'xgb_tuned_model_optimized_f1.pkl', 'scaler.pkl', "
          "and 'label_encoders.pkl' are in the same directory as app.py.")
    model = None
    scaler = None
    label_encoders = None
except Exception as e:
    print(f"An unexpected error occurred while loading files: {e}")
    model = None
    scaler = None
    label_encoders = None

# Define the exact order of features your model expects *before* scaling
# This order MUST match the columns of X when your scaler was fitted in the training script.
# Based on your model code, after dropping 'id' and before label encoding/scaling.
EXPECTED_FEATURES_ORDER_BEFORE_SCALING = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

# Mappings for binary fields (from string to int, as your model implies 0/1 for these)
HYPERTENSION_MAP = {'Yes': 1, 'No': 0}
HEART_DISEASE_MAP = {'Yes': 1, 'No': 0}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if model and preprocessors were loaded successfully
    if model is None or scaler is None or label_encoders is None:
        return jsonify({
            'error': 'Server configuration error.',
            'message': 'Prediction service is not fully initialized. '
                       'Please check server logs for missing model/preprocessor files.'
        }), 500

    try:
        data = request.form.to_dict()
        print("Received form data:", data)

        # 1. Prepare data into a format suitable for the model's original features
        processed_input_dict = {}

        # Handle numerical features (age, avg_glucose_level, bmi)
        processed_input_dict['age'] = float(data['age'])
        processed_input_dict['avg_glucose_level'] = float(data['avg_glucose_level'])
        processed_input_dict['bmi'] = float(data['bmi'])

        # Handle binary features (hypertension, heart_disease)
        processed_input_dict['hypertension'] = HYPERTENSION_MAP.get(data['hypertension'])
        processed_input_dict['heart_disease'] = HEART_DISEASE_MAP.get(data['heart_disease'])

        # Handle categorical features using loaded LabelEncoders
        categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
        for col in categorical_cols:
            le = label_encoders.get(col)
            if le:
                try:
                    # Transform the input string value to its numerical representation
                    processed_input_dict[col] = le.transform([data[col]])[0]
                except ValueError:
                    # This error occurs if the input value is not in the LabelEncoder's known classes
                    return jsonify({
                        'error': f"Invalid input for {col}.",
                        'message': f"The selected value '{data[col]}' for {col} is not recognized. "
                                   "Please select a valid option."
                    }), 400
            else:
                # Should not happen if label_encoders are correctly saved/loaded
                return jsonify({
                    'error': f"LabelEncoder not found for {col}.",
                    'message': "Server preprocessing configuration error."
                }), 500

        # 2. Create a Pandas DataFrame from the processed data
        # Ensure the columns are in the exact order the scaler (and thus the model) expects.
        input_df_before_scaling = pd.DataFrame([processed_input_dict], columns=EXPECTED_FEATURES_ORDER_BEFORE_SCALING)

        print("Input DataFrame before scaling:\n", input_df_before_scaling)

        # 3. Apply the StandardScaler transformation
        # scaler.transform returns a NumPy array.
        input_scaled_array = scaler.transform(input_df_before_scaling)

        # 4. Make prediction using the loaded model
        # The model expects a 2D array, which input_scaled_array is.
        prediction = model.predict(input_scaled_array)[0] # [0] to get the scalar prediction
        prediction_proba = model.predict_proba(input_scaled_array)[:, 1][0] # Probability of the positive class

        # Interpret prediction
        result_message = "High Risk of Stroke" if prediction == 1 else "Low Risk of Stroke"
        probability_message = f"{prediction_proba * 100:.2f}%"

        return jsonify({
            'result': result_message,
            'probability': probability_message
        })

    except ValueError as ve:
        # Catches errors from float() conversion or other data type issues
        return jsonify({
            'error': 'Invalid input data format.',
            'message': f"Please ensure all numerical fields are valid numbers. Error: {str(ve)}"
        }), 400
    except KeyError as ke:
        # Catches errors if a required form field is missing unexpectedly
        return jsonify({
            'error': 'Missing form data.',
            'message': f"A required field is missing or malformed: {str(ke)}. Please ensure all fields are correctly filled."
        }), 400
    except Exception as e:
        # Catch any other unexpected errors during prediction process
        print(f"An unexpected error occurred during prediction: {e}")
        return jsonify({
            'error': 'Prediction failed.',
            'message': f"An internal server error occurred during prediction. Please try again later. ({str(e)})"
        }), 500

if __name__ == '__main__':
    app.run(debug=True) # debug=True is good for development, turn off in production

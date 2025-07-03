from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os # Import os module to handle paths

app = Flask(__name__)

# --- Load the Model and Preprocessing Objects ---
# Determine the base directory of the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the path to the Models directory
MODELS_DIR = os.path.join(BASE_DIR, 'Models')

try:
    model = joblib.load(os.path.join(MODELS_DIR, 'Stroke_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    label_encoders = joblib.load(os.path.join(MODELS_DIR, 'label_encoders.pkl'))
    train_bmi_mean = joblib.load(os.path.join(MODELS_DIR, 'train_bmi_mean.pkl'))
    print("Model and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"Error loading model or preprocessing objects: {e}")
    # Handle error, maybe exit or set app to a 'maintenance mode'
    model = None
    scaler = None
    label_encoders = None
    train_bmi_mean = None

# Define the order of features the model expects (IMPORTANT!)
# This should match the order of columns in X during model training
FEATURE_ORDER = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

# --- Helper Function to Generate Result Content ---
def generate_result_content(stroke_percent, user_data):
    """
    Generates the rich HTML content for the stroke risk assessment.
    """
    risk_level = ""
    risk_badge_class = "" # New: for styling the badge
    risk_message = "" # New: for the specific risk message
    
    recommendations = {
        "contributors": [],
        "clinical_actions": [],
        "lifestyle_recommendations": [],
        "recommended_foods": {
            "Male": [
                {"name": "Lean chicken", "icon": "🍗"},
                {"name": "Brown rice", "icon": "🍚"},
                {"name": "Chia seeds", "icon": "🌱"},
                {"name": "Broccoli", "icon": "🥦"},
                {"name": "Olive oil", "icon": "🫒"},
                {"name": "Turkey", "icon": "🦃"},
                {"name": "Lentils", "icon": "🥣"},
                {"name": "Tomatoes", "icon": "🍅"}
            ],
            "Female": [
                {"name": "Leafy greens", "icon": "🥬"},
                {"name": "Salmon", "icon": "🍣"},
                {"name": "Nuts & Seeds", "icon": "🌰"},
                {"name": "Citrus fruits", "icon": "🍊"},
                {"name": "Beans & Lentils", "icon": "🍲"},
                {"name": "Yogurt", "icon": "🥛"},
                {"name": "Dark chocolate", "icon": "🍫"},
                {"name": "Avocado", "icon": "🥑"}
            ]
        }
    }

    # Determine Risk Level, Badge Class, and Message
    if stroke_percent < 10:
        risk_level = "Very Low Risk"
        risk_badge_class = "risk-green"
        risk_message = "Very low risk detected. Continue your healthy habits!"
    elif 10 <= stroke_percent < 30:
        risk_level = "Moderate Risk"
        risk_badge_class = "risk-yellow"
        risk_message = "Moderate risk detected. Time to take action with lifestyle changes."
    elif 30 <= stroke_percent < 50:
        risk_level = "High Risk"
        risk_badge_class = "risk-orange"
        risk_message = "High risk detected. Consult a healthcare professional."
    elif 50 <= stroke_percent < 70:
        risk_level = "Very High Risk"
        risk_badge_class = "risk-red"
        risk_message = "🚨 Very High risk detected! Immediate medical consultation strongly recommended."
    else: # stroke_percent >= 70
        risk_level = "Extreme Risk"
        risk_badge_class = "risk-red-dark"
        risk_message = "🚨 Extreme risk detected! Urgent medical intervention advised."

    # Main Risk Contributors (Adjusted based on common factors and example)
    if user_data['age'] >= 60:
        recommendations["contributors"].append(f"Age {user_data['age']} - Risk significantly increases with age")
    elif user_data['age'] >= 40:
        recommendations["contributors"].append(f"Age {user_data['age']} - Risk increases with age")
    
    if user_data['hypertension'] == 1:
        recommendations["contributors"].append("Hypertension - High blood pressure is a major risk factor")
    if user_data['heart_disease'] == 1:
        recommendations["contributors"].append("Heart Disease - Pre-existing heart conditions elevate risk")
    
    if user_data['smoking_status_original'] == 'smokes':
        recommendations["contributors"].append("Active Smoker - Smoking severely damages blood vessels and increases clot risk")
    elif user_data['smoking_status_original'] == 'formerly smoked':
        recommendations["contributors"].append("Formerly Smoked - Past smoking history still contributes to risk")
    
    # Placeholder for cholesterol (not an input, but in example result)
    # If you add an input, replace this with dynamic logic.
    recommendations["contributors"].append("Borderline high cholesterol (200-239 mg/dL)") 

    if user_data['bmi'] < 18.5:
        recommendations["contributors"].append("Underweight (BMI <18.5) - May indicate poor nutrition or underlying health issues")
    elif user_data['bmi'] >= 25 and user_data['bmi'] < 30:
        recommendations["contributors"].append("Overweight (BMI 25-29.9) - Increases strain on the cardiovascular system")
    elif user_data['bmi'] >= 30:
        recommendations["contributors"].append("Obese (BMI >30) - Significantly elevates risk for stroke and related conditions")

    if not recommendations["contributors"]: # If no specific contributors found, add a general one
        recommendations["contributors"].append("Based on your inputs, no major individual risk factors are identified, but general health maintenance is always advised.")

    # Clinical Actions (Adapted to new example format)
    recommendations["clinical_actions"].append("Annual checkup - Important for prevention")
    recommendations["clinical_actions"].append("Lipid profile test - Check HDL, LDL, triglycerides")
    if stroke_percent >= 30:
        recommendations["clinical_actions"].append("Consult your primary care physician for a comprehensive health assessment")
        recommendations["clinical_actions"].append("Consider blood pressure and cholesterol management strategies")
    if stroke_percent >= 50:
        recommendations["clinical_actions"].append("Urgent medical consultation - Schedule with a doctor immediately")
        recommendations["clinical_actions"].append("Discuss preventative medication if appropriate.")


    # Lifestyle Recommendations (Adapted to new example format)
    recommendations["lifestyle_recommendations"].append("Exercise regularly - 30 minutes most days")
    recommendations["lifestyle_recommendations"].append("Balanced diet - Focus on whole foods")
    recommendations["lifestyle_recommendations"].append("Maintain healthy weight - BMI 18.5-24.9 ideal")
    recommendations["lifestyle_recommendations"].append("Healthy fats - More nuts, fish, olive oil")
    if user_data['smoking_status_original'] in ['smokes', 'formerly smoked']:
        recommendations["lifestyle_recommendations"].append("Quit smoking immediately - Most important change you can make")
    if stroke_percent >= 30:
        recommendations["lifestyle_recommendations"].append("Stress management - Try meditation or yoga")


    # Construct HTML output using the new card structure
    html_output = f"""
    <div class="result-card result-main">
        <h2 class="result-title">Stroke Risk Assessment</h2>
        <p class="stroke-percent">{stroke_percent:.1f}%</p>
        <div class="risk-badge {risk_badge_class}">
            <span class="icon">{"⚠️" if stroke_percent >= 30 else "🟢"}</span> {risk_level}
        </div>
        <p class="risk-message">
            <span class="icon">{"🔔" if stroke_percent >= 30 else "💡"}</span> {risk_message}
        </p>
    </div>

    <div class="result-card">
        <h3><span class="icon">📌</span> Main Risk Contributors:</h3>
        <ul>
            {''.join([f'<li><span class="bullet-icon">🔹</span> {c}</li>' for c in recommendations['contributors']])}
        </ul>
    </div>

    <div class="result-card">
        <h3><span class="icon">🏥</span> Clinical Actions:</h3>
        <ul>
            {''.join([f'<li><span class="bullet-icon">🩺</span> {c}</li>' for c in recommendations['clinical_actions']])}
        </ul>
    </div>

    <div class="result-card">
        <h3><span class="icon">🌿</span> Lifestyle Recommendations:</h3>
        <ul>
            {''.join([f'<li><span class="bullet-icon">💪</span> {c}</li>' for c in recommendations['lifestyle_recommendations']])}
        </ul>
    </div>
    """
    
    user_gender = user_data['gender_original']
    if user_gender in recommendations['recommended_foods']:
        food_list_items = ''.join([f'<span class="food-tag">{food["name"]} {food["icon"]}</span>' for food in recommendations['recommended_foods'][user_gender]])
        html_output += f"""
        <div class="result-card">
            <h3><span class="icon">🍎</span> Recommended Foods for Your Gender ({user_gender}):</h3>
            <div class="food-tags-container">
                {food_list_items}
            </div>
        </div>
        """
    else: # Fallback if gender is 'Other' or not recognized
        food_list_items = ''.join([f'<span class="food-tag">{food["name"]} {food["icon"]}</span>' for food in recommendations['recommended_foods']['Male']]) # Default to Male foods
        html_output += f"""
        <div class="result-card">
            <h3><span class="icon">🍎</span> Recommended Foods:</h3>
            <p>A balanced diet rich in fruits, vegetables, and whole grains is recommended for everyone.</p>
            <div class="food-tags-container">
                {food_list_items}
            </div>
        </div>
        """

    return html_output


# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or label_encoders is None or train_bmi_mean is None:
        return jsonify({"error": "Model not loaded. Server is not ready."}), 500

    try:
        data = request.form.to_dict()

        # Store original values for result generation
        user_data = {
            'gender_original': data['gender'],
            'age': float(data['age']),
            'hypertension': int(data['hypertension']),
            'heart_disease': int(data['heart_disease']),
            'ever_married': data['ever_married'],
            'work_type': data['work_type'],
            'Residence_type': data['Residence_type'],
            'avg_glucose_level': float(data['avg_glucose_level']),
            'bmi': float(data['bmi']) if data['bmi'] else None, # Handle empty BMI
            'smoking_status_original': data['smoking_status']
        }

        # Preprocessing steps
        # 1. Handle 'Other' gender if it somehow sneaks in (though HTML should prevent it)
        processed_gender = data['gender']
        if processed_gender == 'Other':
            # Map 'Other' to a default like 'Male' or 'Female' or the most frequent from your training
            # For simplicity, if 'Other' is passed, we'll try to map it to 'Male' or raise error.
            # Best practice is to ensure form only sends valid options.
            processed_gender = 'Male'
            print("Warning: 'Other' gender received, mapping to 'Male'. Ensure form restricts valid options.")
        
        # 2. Impute BMI if missing
        if user_data['bmi'] is None:
            user_data['bmi'] = train_bmi_mean
            print(f"BMI was missing, imputed with training mean: {train_bmi_mean:.2f}")

        # 3. Label Encoding for categorical features
        processed_input = {}
        for col, le in label_encoders.items():
            # Check if the value from user_data is in the encoder's known classes
            value_to_encode = None
            if col == 'gender': value_to_encode = processed_gender
            elif col == 'ever_married': value_to_encode = user_data['ever_married']
            elif col == 'work_type': value_to_encode = user_data['work_type']
            elif col == 'Residence_type': value_to_encode = user_data['Residence_type']
            elif col == 'smoking_status': value_to_encode = user_data['smoking_status_original']
            
            if value_to_encode and value_to_encode in le.classes_:
                processed_input[col] = le.transform([value_to_encode])[0]
            else:
                # Handle unknown categories: either map to a default, or raise an error.
                # For simplicity, if a category isn't known, this will raise a ValueError from transform
                # which will be caught by the outer try-except.
                # A more robust solution might map to the most frequent category or 0.
                print(f"Warning: Unknown category '{value_to_encode}' for column '{col}'.")
                processed_input[col] = le.transform([value_to_encode])[0] # Will likely fail here if unknown

        # Add numerical features
        processed_input['age'] = user_data['age']
        processed_input['hypertension'] = user_data['hypertension']
        processed_input['heart_disease'] = user_data['heart_disease']
        processed_input['avg_glucose_level'] = user_data['avg_glucose_level']
        processed_input['bmi'] = user_data['bmi']

        # Create DataFrame from processed input
        # Ensure column order matches the training data
        input_df = pd.DataFrame([processed_input])[FEATURE_ORDER]

        # 4. Scale numerical features
        scaled_input = scaler.transform(input_df)
        
        # 5. Make prediction
        prediction_proba = model.predict_proba(scaled_input)[0][1] # Probability of stroke (class 1)
        stroke_percent = prediction_proba * 100

        # Generate the rich result content
        result_html = generate_result_content(stroke_percent, user_data)

        return jsonify({"success": True, "result_html": result_html})

    except ValueError as ve:
        return jsonify({"error": f"Invalid input or unknown category: {ve}. Please check your inputs."}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# This block is only for local development/testing, Render/Vercel manage the server startup.
if __name__ == '__main__':
    app.run(debug=True)

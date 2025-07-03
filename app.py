from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Load the Model and Preprocessing Objects ---
try:
    model = joblib.load('Models/Stroke_model.pkl') # Updated path
    scaler = joblib.load('Models/scaler.pkl')       # Updated path
    label_encoders = joblib.load('Models/label_encoders.pkl') # Updated path
    train_bmi_mean = joblib.load('Models/train_bmi_mean.pkl') # Updated path
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
# You can get this from X.columns after the initial data loading and ID drop.
# Example (adjust based on your actual X.columns):
# X = df.drop('stroke', axis=1)
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
    risk_icon = ""
    recommendations = {
        "contributors": [],
        "clinical_actions": [],
        "lifestyle_recommendations": [],
        "recommended_foods": {
            "Male": [
                "Boiled vegetables 🥕", "Steamed fish 🐟", "Green tea 🍵",
                "Whole wheat 🥖", "Berries 🍓", "Oats 🥣", "Cucumber 🥒", "Celery 🥬"
            ],
            "Female": [
                "Leafy greens 🥬", "Salmon 🍣 (rich in Omega-3)", "Nuts & Seeds 🌰",
                "Citrus fruits 🍊", "Beans & Lentils 🍲", "Yogurt 🥛 (low-fat)",
                "Dark chocolate 🍫 (in moderation)", "Avocado 🥑"
            ]
        }
    }

    # Determine Risk Level and Icon
    if stroke_percent < 10:
        risk_level = "Very Low Risk"
        risk_icon = "🟢"
        recommendations["clinical_actions"].append("Continue healthy lifestyle. Regular check-ups recommended.")
        recommendations["lifestyle_recommendations"].append("Maintain balanced diet and regular exercise.")
    elif 10 <= stroke_percent < 30:
        risk_level = "Low Risk"
        risk_icon = "🔵"
        recommendations["clinical_actions"].append("Routine medical check-up within 6-12 months.")
        recommendations["lifestyle_recommendations"].append("Focus on a balanced diet and consistent physical activity.")
    elif 30 <= stroke_percent < 50:
        risk_level = "Moderate Risk"
        risk_icon = "🟡"
        recommendations["clinical_actions"].append("Consult your primary care physician for a comprehensive health assessment.")
        recommendations["lifestyle_recommendations"].append("Adopt healthier eating habits and increase physical activity.")
        recommendations["lifestyle_recommendations"].append("Monitor blood pressure and cholesterol levels regularly.")
    elif 50 <= stroke_percent < 70:
        risk_level = "High Risk"
        risk_icon = "🟠 🟠"
        recommendations["clinical_actions"].append("Urgent medical consultation - Schedule with a doctor immediately.")
        recommendations["clinical_actions"].append("Consider blood pressure and cholesterol management strategies.")
        recommendations["lifestyle_recommendations"].append("Implement significant lifestyle changes: quit smoking, dietary improvements, consistent exercise.")
        recommendations["lifestyle_recommendations"].append("Stress management techniques (e.g., meditation, yoga) can be beneficial.")
    else: # stroke_percent >= 70
        risk_level = "Very High Risk"
        risk_icon = "🔴 🔴"
        recommendations["clinical_actions"].append("🚨 High risk detected! Immediate medical consultation strongly recommended. Discuss preventative medication if appropriate.")
        recommendations["clinical_actions"].append("Regular monitoring of vital signs (BP, glucose) is critical.")
        recommendations["clinical_actions"].append("Consider specialist referral (e.g., cardiologist, neurologist).")
        recommendations["lifestyle_recommendations"].append("Aggressive lifestyle changes: strict adherence to dietary guidelines, physician-approved exercise program.")
        recommendations["lifestyle_recommendations"].append("Comprehensive stress reduction plan.")

    # Main Risk Contributors
    if user_data['age'] >= 60:
        recommendations["contributors"].append(f"Age {user_data['age']} - Risk significantly increases with age.")
    elif user_data['age'] >= 40:
        recommendations["contributors"].append(f"Age {user_data['age']} - Risk increases with age.")
    if user_data['hypertension'] == 1:
        recommendations["contributors"].append("Hypertension - High blood pressure is a major risk factor.")
    if user_data['heart_disease'] == 1:
        recommendations["contributors"].append("Heart Disease - Pre-existing heart conditions elevate risk.")
    if user_data['smoking_status_original'] == 'smokes':
        recommendations["contributors"].append("Active Smoker - Smoking severely damages blood vessels and increases clot risk.")
    elif user_data['smoking_status_original'] == 'formerly smoked':
        recommendations["contributors"].append("Formerly Smoked - Past smoking history still contributes to risk.")

    if user_data['bmi'] < 18.5:
        recommendations["contributors"].append("Underweight (BMI <18.5) - May indicate poor nutrition or underlying health issues.")
    elif user_data['bmi'] >= 25 and user_data['bmi'] < 30:
        recommendations["contributors"].append("Overweight (BMI 25-29.9) - Increases strain on the cardiovascular system.")
    elif user_data['bmi'] >= 30:
        recommendations["contributors"].append("Obese (BMI >30) - Significantly elevates risk for stroke and related conditions.")

    if not recommendations["contributors"]: # If no specific contributors found, add a general one
        recommendations["contributors"].append("Based on your inputs, no major individual risk factors are identified, but general health maintenance is always advised.")

    # Add general lifestyle recommendations if not enough specific ones
    if not recommendations["lifestyle_recommendations"]:
        recommendations["lifestyle_recommendations"].append("Maintain a healthy weight.")
        recommendations["lifestyle_recommendations"].append("Regular physical activity (aim for 30 minutes most days).")
        recommendations["lifestyle_recommendations"].append("Eat a balanced diet rich in fruits, vegetables, and whole grains.")
        recommendations["lifestyle_recommendations"].append("Limit saturated and trans fats, cholesterol, and sodium.")

    # Construct HTML output
    html_output = f"""
    <h2>Stroke Risk Assessment</h2>
    <p class="stroke-percent">{stroke_percent:.1f}%</p>
    <p class="risk-level">{risk_icon} {risk_level}</p>
    <p class="call-to-action">
        {recommendations['clinical_actions'][0] if recommendations['clinical_actions'] else 'Maintain a healthy lifestyle.'}
    </p>

    <h3>🔍 Main Risk Contributors:</h3>
    <ul>
        {''.join([f'<li>🔸 {c}</li>' for c in recommendations['contributors']])}
    </ul>

    <h3>🏥 Clinical Actions:</h3>
    <ul>
        {''.join([f'<li>🩺 {c}</li>' for c in recommendations['clinical_actions']])}
    </ul>

    <h3>🌿 Lifestyle Recommendations:</h3>
    <ul>
        {''.join([f'<li>🏋️ {c}</li>' for c in recommendations['lifestyle_recommendations']])}
    </ul>
    """
    
    user_gender = user_data['gender_original']
    if user_gender in recommendations['recommended_foods']:
        html_output += f"""
        <h3>🍎 Recommended Foods for Your Gender ({user_gender}):</h3>
        <ul>
            {''.join([f'<li>{food}</li>' for food in recommendations['recommended_foods'][user_gender]])}
        </ul>
        """
    else:
        html_output += """
        <h3>🍎 Recommended Foods:</h3>
        <p>A balanced diet rich in fruits, vegetables, and whole grains is recommended for everyone.</p>
        <ul>
            <li>Berries 🍓</li>
            <li>Oats 🥣</li>
            <li>Leafy greens 🥬</li>
            <li>Fish 🐟</li>
            <li>Nuts & Seeds 🌰</li>
        </ul>
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
        # Your original code drops 'Other'. For inference, we can map it or handle as error.
        # For this setup, we'll assume the form restricts to 'Male'/'Female'.
        processed_gender = data['gender']
        if processed_gender == 'Other':
            # Option 1: Map 'Other' to a default like 'Male' or 'Female' or the most frequent
            # For simplicity, let's assume 'Male' if 'Other' is somehow passed
            processed_gender = 'Male' # Or raise error if strict validation needed
            print("Warning: 'Other' gender received, mapping to 'Male'.")
        
        # 2. Impute BMI if missing
        if user_data['bmi'] is None:
            user_data['bmi'] = train_bmi_mean
            print(f"BMI was missing, imputed with training mean: {train_bmi_mean:.2f}")

        # 3. Label Encoding for categorical features
        processed_input = {}
        for col, le in label_encoders.items():
            if col == 'gender':
                processed_input[col] = le.transform([processed_gender])[0]
            elif col == 'ever_married':
                processed_input[col] = le.transform([user_data['ever_married']])[0]
            elif col == 'work_type':
                processed_input[col] = le.transform([user_data['work_type']])[0]
            elif col == 'Residence_type':
                processed_input[col] = le.transform([user_data['Residence_type']])[0]
            elif col == 'smoking_status':
                processed_input[col] = le.transform([user_data['smoking_status_original']])[0]
        
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
        return jsonify({"error": f"Invalid input: {ve}. Please check your numerical values."}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True) # Set debug=False in production

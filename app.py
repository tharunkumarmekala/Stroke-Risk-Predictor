from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Load the Model and Preprocessing Objects ---
try:
    model = joblib.load('Models/Stroke_model.pkl')
    scaler = joblib.load('Models/scaler.pkl')
    label_encoders = joblib.load('Models/label_encoders.pkl')
    train_bmi_mean = joblib.load('Models/train_bmi_mean.pkl')
    print("Model and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"Error loading model or preprocessing objects: {e}")
    # Handle error, maybe exit or set app to a 'maintenance mode'
    model = None
    scaler = None
    label_encoders = None
    train_bmi_mean = None

# Define the order of features the model expects (IMPORTANT!)
FEATURE_ORDER = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
]

# --- Helper Function to Generate Result Content ---
def generate_result_content(stroke_percent, user_data):
    """
    Generates the rich HTML content for the stroke risk assessment.
    """
    risk_level_text = ""
    risk_icon = ""
    call_to_action = ""

    # Determine Risk Level and Action
    if stroke_percent < 10:
        risk_level_text = "Very Low Risk"
        risk_icon = "🟢"
        call_to_action = "Great! Continue your healthy habits for long-term well-being."
    elif 10 <= stroke_percent < 30:
        risk_level_text = "Low Risk"
        risk_icon = "🔵"
        call_to_action = "Good news! Maintain a healthy lifestyle and regular check-ups."
    elif 30 <= stroke_percent < 50:
        risk_level_text = "Moderate Risk"
        risk_icon = "🟡"
        call_to_action = "Moderate risk detected. Time to take action with lifestyle changes."
    elif 50 <= stroke_percent < 70:
        risk_level_text = "High Risk"
        risk_icon = "🟠"
        call_to_action = "High risk detected! Medical consultation strongly recommended."
    else: # stroke_percent >= 70
        risk_level_text = "Very High Risk"
        risk_icon = "🔴"
        call_to_action = "🚨 Very high risk detected! Immediate medical attention is crucial."

    # Initialize recommendation lists
    main_risk_contributors = []
    clinical_actions = []
    lifestyle_recommendations = []
    recommended_foods = {
        "Male": [
            "Lean chicken 🍗", "Brown rice 🍚", "Chia seeds 🌱", "Broccoli 🥦",
            "Olive oil 🫒", "Turkey 🦃", "Lentils 🥣", "Tomatoes 🍅"
        ],
        "Female": [
            "Salmon 🍣", "Spinach 🥬", "Avocado 🥑", "Quinoa 🍚",
            "Blueberries 🫐", "Greek Yogurt 🥛", "Almonds 🌰", "Sweet Potatoes 🍠"
        ]
    }

    # Populate Risk Contributors
    if user_data['age'] >= 60:
        main_risk_contributors.append(f"👴 Age {user_data['age']} - Risk significantly increases with age.")
    elif user_data['age'] >= 40:
        main_risk_contributors.append(f"👨‍🦰 Age {user_data['age']} - Risk increases with age.")

    if user_data['hypertension'] == 1:
        main_risk_contributors.append("🩸 Hypertension - High blood pressure is a major risk factor.")
        clinical_actions.append("🩺 Blood pressure monitoring - Regular checks are vital.")
        lifestyle_recommendations.append("🧂 Limit sodium intake - Reduce processed foods and salt.")
    
    if user_data['heart_disease'] == 1:
        main_risk_contributors.append("🫀 Heart Disease - Pre-existing heart conditions elevate risk.")
        clinical_actions.append("❤️ Cardiac evaluation - Consult a cardiologist.")
        lifestyle_recommendations.append("🏃‍♀️ Start supervised exercise - Begin with light activity daily.")

    # General cholesterol recommendation (since model doesn't output it)
    # This is a heuristic based on generally unhealthy parameters or higher risk.
    if user_data['avg_glucose_level'] > 120 or user_data['bmi'] > 27 or user_data['hypertension'] == 1:
        main_risk_contributors.append("🌡️ Borderline high cholesterol (200-239 mg/dL) - May indicate cardiovascular strain.")
        clinical_actions.append("🔬 Lipid profile test - Check HDL, LDL, triglycerides.")
        lifestyle_recommendations.append("🥑 Healthy fats - More nuts, fish, olive oil.")


    if user_data['smoking_status_original'] == 'smokes':
        main_risk_contributors.append("🚬 Active Smoker - Smoking severely damages blood vessels and increases clot risk.")
        clinical_actions.append("🚭 Smoking cessation support - Seek help to quit immediately.")
        lifestyle_recommendations.append("🛑 Quit smoking immediately - Most important change you can make.")
    elif user_data['smoking_status_original'] == 'formerly smoked':
        main_risk_contributors.append("🚭 Formerly Smoked - Past smoking history still contributes to risk.")

    bmi_status = ""
    if user_data['bmi'] < 18.5:
        bmi_status = "Underweight"
        main_risk_contributors.append("⬇️ Underweight (BMI <18.5) - May indicate poor nutrition or underlying health issues.")
        lifestyle_recommendations.append("🍽️ Nutritional consultation - Discuss healthy weight gain strategies.")
    elif user_data['bmi'] >= 25 and user_data['bmi'] < 30:
        bmi_status = "Overweight"
        main_risk_contributors.append("⬆️ Overweight (BMI 25-29.9) - Increases strain on the cardiovascular system.")
        lifestyle_recommendations.append("⚖️ Maintain healthy weight - BMI 18.5-24.9 ideal.")
    elif user_data['bmi'] >= 30:
        bmi_status = "Obese"
        main_risk_contributors.append("📈 Obese (BMI >30) - Significantly elevates risk for stroke and related conditions.")
        lifestyle_recommendations.append("⚖️ Achieve healthy weight - Work towards a healthy BMI.")

    if not main_risk_contributors:
        main_risk_contributors.append("✅ No specific major risk factors identified based on your inputs. Continue to monitor your health.")

    # General Clinical Actions
    if not any("Annual checkup" in s for s in clinical_actions): # Avoid duplicates
        clinical_actions.insert(0, "🩺 Annual checkup - Important for prevention")

    # General Lifestyle Recommendations
    if not any("Exercise" in s for s in lifestyle_recommendations):
        lifestyle_recommendations.append("🏃‍♀️ Exercise regularly - 30 minutes most days")
    if not any("Balanced diet" in s for s in lifestyle_recommendations):
        lifestyle_recommendations.append("🥗 Balanced diet - Focus on whole foods")
    if not any("Stress management" in s for s in lifestyle_recommendations):
        lifestyle_recommendations.append("🧘 Stress management - Try meditation or yoga")

    # Construct HTML output for results
    html_output = f"""
    <div class="result-card result-summary">
        <h2>Stroke Risk Assessment</h2>
        <p class="stroke-percent">{stroke_percent:.1f}%</p>
        <p class="risk-level">{risk_icon} {risk_level_text}</p>
        <p class="call-to-action">
            {risk_icon} {call_to_action}
        </p>
    </div>

    <div class="result-card">
        <h3><span class="icon">🔍</span> Main Risk Contributors:</h3>
        <ul>
            {''.join([f'<li>{c}</li>' for c in main_risk_contributors])}
        </ul>
    </div>

    <div class="result-card">
        <h3><span class="icon">🏥</span> Clinical Actions:</h3>
        <ul>
            {''.join([f'<li>{c}</li>' for c in clinical_actions])}
        </ul>
    </div>

    <div class="result-card">
        <h3><span class="icon">🌿</span> Lifestyle Recommendations:</h3>
        <ul>
            {''.join([f'<li>{c}</li>' for c in lifestyle_recommendations])}
        </ul>
    </div>
    """
    
    user_gender = user_data['gender_original']
    if user_gender in recommended_foods:
        html_output += f"""
        <div class="result-card">
            <h3><span class="icon">🍎</span> Recommended Foods for Your Gender ({user_gender}):</h3>
            <div class="food-pills">
                {''.join([f'<span class="food-pill">{food}</span>' for food in recommended_foods[user_gender]])}
            </div>
        </div>
        """
    else:
        # Fallback for "Other" gender or if gender isn't in list
        html_output += """
        <div class="result-card">
            <h3><span class="icon">🍎</span> Recommended Foods:</h3>
            <p>A balanced diet rich in fruits, vegetables, and whole grains is recommended for everyone.</p>
            <div class="food-pills">
                <span class="food-pill">Berries 🍓</span>
                <span class="food-pill">Oats 🥣</span>
                <span class="food-pill">Leafy greens 🥬</span>
                <span class="food-pill">Fish 🐟</span>
                <span class="food-pill">Nuts & Seeds 🌰</span>
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

        # Extract data from new form structure
        # Use .get() for optional fields like BMI to avoid KeyError if empty
        user_data = {
            'gender_original': data['gender'],
            'age': float(data['age']),
            'hypertension': int(data['hypertension']),
            'heart_disease': int(data['heart_disease']),
            'ever_married': data['ever_married'],
            'work_type': data['work_type'],
            'Residence_type': data['Residence_type'],
            'avg_glucose_level': float(data['avg_glucose_level']),
            'bmi': float(data.get('bmi')) if data.get('bmi') else None, # Handle empty BMI
            'smoking_status_original': data['smoking_status']
        }

        # Preprocessing steps
        processed_gender = user_data['gender_original']
        if processed_gender == 'Other':
            # Handle 'Other' gender as per your model's training behavior
            # Your original code dropped 'Other', so here we'll map it to 'Female' as a practical choice
            # or you could raise an error if strict input validation is preferred.
            processed_gender = 'Female' # or 'Male', based on which is more representative or frequent
            print("Warning: 'Other' gender received, mapping to 'Female' for prediction.")

        # Impute BMI if missing
        if user_data['bmi'] is None:
            user_data['bmi'] = train_bmi_mean
            print(f"BMI was missing, imputed with training mean: {train_bmi_mean:.2f}")

        # Label Encoding for categorical features
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
        input_df = pd.DataFrame([processed_input])[FEATURE_ORDER]

        # Scale numerical features
        scaled_input = scaler.transform(input_df)
        
        # Make prediction
        prediction_proba = model.predict_proba(scaled_input)[0][1] # Probability of stroke (class 1)
        stroke_percent = prediction_proba * 100

        # Generate the rich result content
        result_html = generate_result_content(stroke_percent, user_data)

        return jsonify({"success": True, "result_html": result_html})

    except ValueError as ve:
        return jsonify({"error": f"Invalid input: {ve}. Please check your numerical values."}), 400
    except Exception as e:
        import traceback
        traceback.print_exc() # Print full traceback to console for debugging
        return jsonify({"error": f"An unexpected server error occurred. Please try again. ({e})"}), 500

# Remember to remove app.run(debug=True) for production deployments on Render/Vercel
# if __name__ == '__main__':
#     app.run(debug=True)

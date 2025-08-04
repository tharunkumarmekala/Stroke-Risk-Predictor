# Stroke Risk Predictor

## Overview

The **Stroke Risk Predictor** is a real-time, web-based application that uses machine learning to predict the likelihood of a stroke. The tool leverages **XGBoost**, a powerful gradient boosting algorithm, and **SHAP** (SHapley Additive exPlanations) for model explainability. The system is deployed using the **Flask** web framework, providing an intuitive user interface to input clinical data and receive personalized stroke risk predictions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation Instructions](#installation-instructions)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Real-time Stroke Risk Prediction**: Users can input clinical data (age, BMI, blood pressure, etc.) and get real-time stroke risk predictions.
- **Personalized Health Recommendations**: The system provides tailored lifestyle recommendations based on the user's risk category (low, medium, high).
- **Model Explainability**: SHAP values are used to explain the influence of various features on the prediction, enhancing transparency.
- **Flask-based Web Application**: Easy-to-use web interface for both clinicians and patients to interact with the model.

---

## Installation Instructions

### 1. Clone the repository:

```bash
git clone https://github.com/tharunkumarmekala/StrokeRiskPredictor.git
cd StrokeRiskPredictor
```

### 2. Create a virtual environment (optional but recommended):

```bash
python3 -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Run the Flask application:

```bash
python app.py
```

### Access the web app:
Once the application is running, open a web browser and go to:
http://127.0.0.1:5000

### Input clinical data:
Enter the patient's details such as age, BMI, blood pressure, etc. and submit the form.

### View the prediction:
The system will display a stroke risk percentage along with the risk category (low, medium, high).

### Personalized recommendations:
Based on the risk level, the system will suggest lifestyle changes (e.g., improved diet, exercise, etc.).

## Technologies Used

**Flask**: Web framework for real-time application deployment.

**XGBoost**: Gradient boosting model used for prediction.

**SHAP**: Model interpretability using SHapley values.

**HTML/CSS**: For the web interface.

**JavaScript**: For interactivity and real-time updates.

## Contributing

We welcome contributions to improve the project. If you would like to contribute, please fork the repository, make your changes, and submit a pull request.

### Steps to Contribute:
1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make changes and commit them (`git commit -am 'Add new feature'`)
4. Push the changes to your forked repo (`git push origin feature-branch`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.




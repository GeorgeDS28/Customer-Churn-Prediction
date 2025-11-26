Customer Churn Prediction
What this project does

This repository builds a small, end-to-end machine learning pipeline and simple web interface to predict whether a Telco customer will churn (i.e., stop using the service).
It includes data preprocessing, model training, model artifact saving, and a Flask web app to run live predictions from user input.

Project overview (high level)

Input: customer features from the Telco dataset (demographics, account info, services, tenure, charges).

Process: clean & preprocess data, convert categorical features, train a classifier, evaluate performance, and save the trained model + preprocessing objects.

Output: a binary churn prediction (Yes/No) returned by a web UI or API endpoint.

This project is useful for learning:

Data cleaning & preprocessing

Feature encoding and scaling

Model training, evaluation, and persistence

Serving a model through a Flask web app

File / folder structure — explained step by step

app.py
Web application entry point (Flask).
Responsibilities:

Loads saved model artifacts (model, encoder/scaler, feature list).

Renders templates/index.html for a human-friendly form.

Provides an API endpoint (e.g., /predict) that accepts form data or JSON and returns the prediction.

Converts incoming input into the same preprocessed format the model expects and runs .predict() or .predict_proba().

customer_churn_prediction.py
Training + evaluation script.
Responsibilities:

Loads WA_Fn-UseC_-Telco-Customer-Churn.csv.

Performs exploratory cleaning: handle missing values, fix data types, remove duplicates if any.

Encodes categorical variables (OneHotEncoding / LabelEncoding / Ordinal as appropriate).

Scales numerical features where necessary (StandardScaler or MinMaxScaler).

Splits data into train / test sets.

Trains a classifier (e.g., Logistic Regression, RandomForest, XGBoost).

Evaluates performance (accuracy, precision, recall, F1, confusion matrix, AUC).

Saves artifacts: trained model (pickle/joblib), encoders/scalers, and a JSON/py file listing feature order.

WA_Fn-UseC_-Telco-Customer-Churn.csv
The Telco Customer Churn dataset. Contains customer records with churn label.
Notes:

Publicly available dataset — check original license before redistribution.

Typical columns: customerID, gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn.

input.txt (optional)
Example raw input (flat key:value pairs) used by the app/script for quick local testing (useful for CLI or simple API calls).

templates/index.html
HTML template used by Flask to render a simple form for entering customer fields and getting a prediction.
It should:

Present friendly labels for each feature (dropdowns for categorical fields, sliders/textboxes for numeric).

Submit a POST to /predict (or whichever endpoint app.py defines).

Display the predicted class and optionally probability.

requirements.txt
Python package list required to run this project. Typical packages:

flask

pandas

scikit-learn

joblib (or pickle)

optional: xgboost, flask-cors, gunicorn
Pin versions if you want reproducibility.

Step-by-step setup & usage
1. Create and activate a virtual environment (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1


macOS / Linux:

python3 -m venv venv
source venv/bin/activate

2. Install dependencies
pip install -r requirements.txt

3. Train the model (generate artifacts)

Run the training script to preprocess the data, train a model, evaluate it, and save artifacts (model + preprocessors).

python customer_churn_prediction.py


What to expect from this script:

Console logs about data shapes, missing values handled, and model metrics.

Output files like model.joblib, encoder.joblib, scaler.joblib, and features.json (names depend on your implementation).

4. Run the web app
python app.py


The app usually starts on http://127.0.0.1:5000/. Open that in your browser to use the form-based UI.

The app’s /predict endpoint will consume form data (POST) or JSON.

Example API usage (curl)

If app.py exposes a JSON API at /predict, you can call it like this:

curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "InternetService": "DSL",
    "Contract": "Month-to-month",
    "MonthlyCharges": 70.5,
    "TotalCharges": 700.0
  }'


Response (example):

{
  "prediction": "Yes",
  "probability": 0.82
}


If your app uses form POST, use the HTML UI or curl -F.

Model & preprocessing details — what typically happens inside customer_churn_prediction.py

Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

Data cleaning

Convert TotalCharges to numeric (some entries may be blank -> NaN) and fill or drop appropriately.

Fill missing values or remove rows with incomplete crucial fields.

Convert SeniorCitizen to int/boolean.

Feature selection / engineering

Drop customerID (identifier not useful for prediction).

Create tenure buckets (optional): e.g., 0-12, 12-24, 24+.

Create interaction features if helpful (e.g., has_internet = InternetService != "No").

Encode categorical variables

For nominal categories: OneHotEncoder or pd.get_dummies.

For binary categories: map Yes/No to 1/0.

Save encoders so the same mapping is applied at inference time.

Scale numeric variables

Use StandardScaler or MinMaxScaler for MonthlyCharges, TotalCharges, tenure.

Save scaler for inference.

Train/test split

train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

Model training

Baseline: Logistic Regression.

Stronger: RandomForestClassifier or XGBoost for better performance.

Tune hyperparameters with GridSearchCV (optional).

Evaluation

Print metrics: accuracy, precision, recall, F1, ROC AUC.

Plot or print confusion matrix.

Save artifacts

joblib.dump(model, "model.joblib")

joblib.dump(preprocessor, "preprocessor.joblib") OR save separate encoder & scaler.

Save a features.json with the exact ordered list of features the model expects.

How app.py should use saved artifacts (high level)

Load model.joblib and preprocessor.joblib (or encoder + scaler).

When a request arrives:

Parse input (form or JSON).

Build a DataFrame or array with the same column order saved in features.json.

Apply the saved encoder/scaler/pipeline to the input.

Call model.predict() and optionally model.predict_proba().

Return a JSON response or render the result on index.html.

Example minimal app.py flow (pseudo)
from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
app = Flask(__name__)

model = joblib.load("model.joblib")
preprocessor = joblib.load("preprocessor.joblib")
feature_order = joblib.load("features.joblib")  # or JSON

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or request.form.to_dict()
    df = pd.DataFrame([data], columns=feature_order)
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0,1]
    pred = "Yes" if prob > 0.5 else "No"
    return jsonify({"prediction": pred, "probability": float(prob)})

Troubleshooting (common issues & fixes)

Server won’t start: check port conflicts; ensure Flask is installed.

Model artifact not found: run training script first and confirm the artifact filenames/paths.

Input mismatch error: ensure the input dictionary keys match feature names and ordering from training.

Different categories: during inference, unseen categorical values will break encoders — handle with handle_unknown="ignore" in scikit-learn encoders, or map unseen values to an “other” bucket.

Numeric parsing errors: ensure strings like TotalCharges are converted to numeric during training and the same conversion is used at inference.

Suggestions / Next steps (improvements)

Add a Dockerfile and docker-compose.yml for reproducible deployment.

Wrap preprocessing + model in a single scikit-learn Pipeline and save that pipeline only (simpler inference).

Add unit tests for preprocessing & API.

Add CI to retrain automatically when data updates.

Add model explainability: SHAP values to show why a user was predicted to churn.

Add a simple frontend dashboard with examples and dataset stats.

Data & License

WA_Fn-UseC_-Telco-Customer-Churn.csv is commonly available on Kaggle. Confirm the dataset license before redistribution.

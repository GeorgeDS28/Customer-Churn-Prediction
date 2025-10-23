from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
loaded_model = model_data["model"]
feature_names = model_data["features_names"]

# Load the encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Collect data from form
    input_data = {
        'gender': request.form['gender'],
        'SeniorCitizen': int(request.form['SeniorCitizen']),
        'Partner': request.form['Partner'],
        'Dependents': request.form['Dependents'],
        'tenure': float(request.form['tenure']),
        'PhoneService': request.form['PhoneService'],
        'MultipleLines': request.form['MultipleLines'],
        'InternetService': request.form['InternetService'],
        'OnlineSecurity': request.form['OnlineSecurity'],
        'OnlineBackup': request.form['OnlineBackup'],
        'DeviceProtection': request.form['DeviceProtection'],
        'TechSupport': request.form['TechSupport'],
        'StreamingTV': request.form['StreamingTV'],
        'StreamingMovies': request.form['StreamingMovies'],
        'Contract': request.form['Contract'],
        'PaperlessBilling': request.form['PaperlessBilling'],
        'PaymentMethod': request.form['PaymentMethod'],
        'MonthlyCharges': float(request.form['MonthlyCharges']),
        'TotalCharges': float(request.form['TotalCharges'])
    }

    # Convert to DataFrame
    input_data_df = pd.DataFrame([input_data])

    # Encode categorical features using saved encoders
    feature_columns_to_encode = [col for col in encoders.keys() if col != 'Churn']
    for column in feature_columns_to_encode:
        if column in input_data_df.columns:
            input_data_df[column] = encoders[column].transform(input_data_df[column])

    # Predict
    prediction = loaded_model.predict(input_data_df)
    pred_prob = loaded_model.predict_proba(input_data_df)[0][1]  # Probability of churn

    # Format result
    result = "Churn" if prediction[0] == 1 else "No Churn"
    return render_template("index.html", prediction_text=f"Prediction: {result}", prob_text=f"Churn Probability: {pred_prob:.2f}")

if __name__ == "__main__":
    app.run(debug=True)

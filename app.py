from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# FIX PATH
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Loan Default Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([data])

    for col in model.named_steps["preprocessor"].feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    df = df[model.named_steps["preprocessor"].feature_names_in_]

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return jsonify({
        "prediction": int(pred),
        "default_probability": float(prob)
    })

if __name__ == "__main__":
    app.run(debug=True)

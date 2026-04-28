import shap
import pickle
import pandas as pd

model = pickle.load(open("model.pkl", "rb"))

def explain_sample(data):
    explainer = shap.Explainer(model.named_steps["model"])
    shap_values = explainer(data)

    shap.summary_plot(shap_values, data) 
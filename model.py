import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from drift import calculate_psi, calculate_ks, calculate_csi

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(r"D:\Loan_Default\data\Givaji Pravalika - Dataset.csv", low_memory=False)
df.drop(columns=["ID"], inplace=True)

# FIX NUMERIC TYPES
numeric_fix_cols = ['Client_Income', 'Credit_Amount', 'Loan_Annuity']
for col in numeric_fix_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df['loan_income_ratio'] = df['Credit_Amount'] / (df['Client_Income'] + 1)
df['annuity_income_ratio'] = df['Loan_Annuity'] / (df['Client_Income'] + 1)

y = df["Default"]
X = df.drop("Default", axis=1)

# -----------------------------
# DROP HIGH CARDINALITY
# -----------------------------
cat_cols = X.select_dtypes(exclude=np.number).columns
high_card_cols = [col for col in cat_cols if X[col].nunique() > 50]
X = X.drop(columns=high_card_cols)

# -----------------------------
# OUTLIERS
# -----------------------------
def cap_outliers(df, cols):
    df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df.loc[:, col] = np.clip(df[col], lower, upper)
    return df

num_cols = X.select_dtypes(include=np.number).columns
X[num_cols] = cap_outliers(X[num_cols], num_cols)

# -----------------------------
# PIPELINE
# -----------------------------
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# -----------------------------
# SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# -----------------------------
# IMBALANCE
# -----------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_p, y_train)

mlflow.set_experiment("Loan Default Prediction")

# =============================
# LOG FUNCTION (IMPROVED)
# =============================
def log_all(model, name):
    with mlflow.start_run(run_name=name):

        model.fit(X_train_res, y_train_res)

        prob = model.predict_proba(X_test_p)[:, 1]

        # 🔥 THRESHOLD TUNING
        threshold = 0.3
        preds = (prob > threshold).astype(int)

        # Metrics
        mlflow.log_metric("auc", roc_auc_score(y_test, prob))
        mlflow.log_metric("precision", precision_score(y_test, preds))
        mlflow.log_metric("recall", recall_score(y_test, preds))
        mlflow.log_metric("f1_score", f1_score(y_test, preds))
        mlflow.log_param("threshold", threshold)

        # Drift
        X_train_base = X_train_p[:, 0]
        X_test_base = X_test_p[:, 0]

        psi = calculate_psi(X_train_base, X_test_base)
        ks = calculate_ks(X_train_base, X_test_base)

        min_len = min(len(X_train_base), len(X_test_base))
        csi = calculate_csi(X_train_base[:min_len], X_test_base[:min_len])

        mlflow.log_metric("PSI", psi)
        mlflow.log_metric("KS", ks)
        mlflow.log_metric("CSI", csi)

        # SHAP
        X_sample = X_test_p[:200].toarray() if hasattr(X_test_p, "toarray") else X_test_p[:200]

        if isinstance(model, XGBClassifier):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)

        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")
        plt.close()

        mlflow.sklearn.log_model(model, "model")

# -----------------------------
# MODELS
# -----------------------------

# 1️⃣ Logistic (baseline)
log_all(LogisticRegression(max_iter=1000), "Logistic")

# 2️⃣ Logistic L1 (feature selection 🔥)
log_all(LogisticRegression(penalty="l1", solver="liblinear"), "Logistic L1")

# 3️⃣ XGBoost (tuned 🔥)
log_all(XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=3   # 🔥 imbalance handling
), "XGBoost Tuned")

# -----------------------------
# FINAL MODEL SAVE
# -----------------------------
final_model = XGBClassifier(n_estimators=500, max_depth=7)
final_model.fit(X_train_res, y_train_res)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", final_model)
])

pickle.dump(pipeline, open("model.pkl", "wb"))

print("✅ FINAL IMPROVED MODEL READY")
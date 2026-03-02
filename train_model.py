import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
df = pd.read_csv("data/student_performance.csv")
print(f"Loaded: {df.shape[0]:,} rows | Columns: {df.columns.tolist()}")

# ─────────────────────────────────────────────
# 2. IQR OUTLIER REMOVAL  (original logic — unchanged)
# ─────────────────────────────────────────────
cols_iqr = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
    "total_score",
]
for col in cols_iqr:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

print(f"After IQR outlier removal: {df.shape[0]:,} rows")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
# Grade → pass/fail
df["result"] = df["grade"].apply(lambda g: 1 if g in ["A", "B", "C"] else 0)

vc = df["result"].value_counts()
print(f"Class distribution → Pass: {vc.get(1,0):,} ({vc.get(1,0)/len(df)*100:.1f}%)  "
      f"Fail: {vc.get(0,0):,} ({vc.get(0,0)/len(df)*100:.1f}%)")

# ── Only 3 behavioral features — NO total_score ──────────────
# total_score is the outcome (grade comes from it), so including it
# would let the model "cheat" and make individual predictions useless.
FEATURE_COLS = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
]

X_raw = df[FEATURE_COLS].copy()
y     = df["result"].copy()

# Clip at 1st/99th percentile (extra safety after IQR)
for col in FEATURE_COLS:
    X_raw[col] = X_raw[col].clip(X_raw[col].quantile(0.01), X_raw[col].quantile(0.99))

# RobustScaler → PolynomialFeatures(degree=2)
scaler        = RobustScaler()
X_scaled      = scaler.fit_transform(X_raw)

poly          = PolynomialFeatures(degree=2, include_bias=False)
X_poly        = poly.fit_transform(X_scaled)
feature_names = poly.get_feature_names_out(FEATURE_COLS)

# ─────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

# ─────────────────────────────────────────────
# 5. SMOTE — balance classes in training set
# ─────────────────────────────────────────────
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

vc_sm = pd.Series(y_train_sm).value_counts()
print(f"After SMOTE → Pass: {vc_sm.get(1,0):,}  Fail: {vc_sm.get(0,0):,}")

# ─────────────────────────────────────────────
# 6. TRAIN MODELS AND COMPARE USING GridSearchCV
# ─────────────────────────────────────────────
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

candidates = {
    "LogisticRegression": {
        "model": LogisticRegression(class_weight="balanced", random_state=42, max_iter=2000),
        "params": {
            "C": [0.1, 1.0, 10.0]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [6, 12]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [3, 6],
            "learning_rate": [0.05, 0.1]
        }
    }
}

print("\n{:<20} {:>10} {:>12} {:>10} {:>12}".format(
    "Model", "Accuracy", "Precision", "Recall", "F1 (weighted)"))
print("-" * 66)

best_name, best_model, best_f1 = None, None, -1
best_params = {}

for name, config in candidates.items():
    print(f"\nRunning GridSearchCV for {name}...")
    grid_search = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        scoring="f1_weighted",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1
    )
    
    # Train grid search
    grid_search.fit(X_train_sm, y_train_sm)
    best_estimator = grid_search.best_estimator_
    print(f"Best params for {name}: {grid_search.best_params_}")
    
    y_pred = best_estimator.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec  = recall_score(y_test, y_pred, average="weighted")
    f1   = f1_score(y_test, y_pred, average="weighted")

    print("{:<20} {:>10.4f} {:>12.4f} {:>10.4f} {:>12.4f}".format(
        name, acc, prec, rec, f1))

    if f1 > best_f1:
        best_f1, best_name, best_model, best_params = f1, name, best_estimator, grid_search.best_params_

print(f"\n🏆 Best overall model: {best_name}  (F1 = {best_f1:.4f})")
print(f"Winning hyperparameters: {best_params}")

# ─────────────────────────────────────────────
# 7. FINAL EVALUATION ON TEST SET
# ─────────────────────────────────────────────
y_pred_final = best_model.predict(X_test)

acc_final  = accuracy_score(y_test, y_pred_final)
prec_final = precision_score(y_test, y_pred_final, average="weighted")
rec_final  = recall_score(y_test, y_pred_final, average="weighted")
f1_final   = f1_score(y_test, y_pred_final, average="weighted")

print("\n── Final Metrics (" + best_name + ") ──────────────────")
print(f"  Accuracy  : {acc_final:.4f}")
print(f"  Precision : {prec_final:.4f}")
print(f"  Recall    : {rec_final:.4f}")
print(f"  F1 Score  : {f1_final:.4f}")

print("\n── Classification Report ──────────────────────────────")
print(classification_report(y_test, y_pred_final, target_names=["Fail", "Pass"]))

cm = confusion_matrix(y_test, y_pred_final, labels=[0, 1])
print("── Confusion Matrix ───────────────────────────────────")
print(f"  TN (correct fail) : {cm[0,0]:>8,}")
print(f"  FP (fail→pass)    : {cm[0,1]:>8,}")
print(f"  FN (pass→fail)    : {cm[1,0]:>8,}")
print(f"  TP (correct pass) : {cm[1,1]:>8,}")

# ─────────────────────────────────────────────
# 8. K-MEANS CLUSTERING (on 3 behavioral features)
# ─────────────────────────────────────────────
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
kmeans.fit(scaler.transform(df[FEATURE_COLS]))

# ─────────────────────────────────────────────
# 9. SAVE MODEL ARTIFACTS
# ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(best_model,             "models/model.pkl")
joblib.dump(scaler,                 "models/scaler.pkl")
joblib.dump(poly,                   "models/poly.pkl")
joblib.dump(kmeans,                 "models/kmeans.pkl")
joblib.dump(feature_names.tolist(), "models/feature_names.pkl")
joblib.dump(FEATURE_COLS,           "models/input_feature_cols.pkl")
joblib.dump({
    "best_model_name": best_name,
    "accuracy":        acc_final,
    "precision":       prec_final,
    "recall":          rec_final,
    "f1":              f1_final,
}, "models/meta.pkl")

print("\n✅ All artifacts saved to models/")
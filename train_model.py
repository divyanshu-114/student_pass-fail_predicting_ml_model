import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE

# ─────────────────────────────
# FIXED PATHS
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────
df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)

# ─────────────────────────────
# 2. HANDLE NULL VALUES
# ─────────────────────────────
df = df.fillna(df.median(numeric_only=True))

# ─────────────────────────────
# 3. OUTLIER REMOVAL (IQR)
# ─────────────────────────────
num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

print("After outlier removal:", df.shape)

# ─────────────────────────────
# 4. PASS/FAIL TARGET
# ─────────────────────────────
# Pass/Fail based on the SAME conditions used in the dashboard:
#   study hours >= 10  AND  attendance >= 75  AND  participation >= 5
df["pass_fail"] = (
    (df["weekly_self_study_hours"] >= 10) &
    (df["attendance_percentage"]   >= 75) &
    (df["class_participation"]     >= 5)
).astype(int)

print("Pass/Fail distribution:\n", df["pass_fail"].value_counts())

# ─────────────────────────────
# 5. LABEL ENCODING FOR GRADES (A–E)
# ─────────────────────────────
grade_encoder = LabelEncoder()
df["grade_encoded"] = grade_encoder.fit_transform(df["grade"])

# Save encoder for decoding later in app
os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(grade_encoder, os.path.join(MODELS_DIR, "grade_encoder.pkl"))

# ─────────────────────────────
# 6. FEATURES
# ─────────────────────────────
FEATURES = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation"
]

X = df[FEATURES]

# ─────────────────────────────
# 7. SCALE 0–1
# ─────────────────────────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ─────────────────────────────
# 8. PASS/FAIL MODEL
# ─────────────────────────────
y_pass = df["pass_fail"]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_pass, test_size=0.2, random_state=42, stratify=y_pass
)

sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

pass_model = RandomForestClassifier(random_state=42)
pass_model.fit(X_train_sm, y_train_sm)

# Evaluation
pred = pass_model.predict(X_test)
print("\nAccuracy :", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))

# ─────────────────────────────
# 9. GRADE PREDICTION MODEL
# ─────────────────────────────
y_grade = df["grade_encoded"]

grade_model = RandomForestClassifier(random_state=42)
grade_model.fit(X_scaled, y_grade)

# ─────────────────────────────
# 10. CLUSTERING
# ─────────────────────────────
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# ─────────────────────────────
# 11. SAVE MODELS
# ─────────────────────────────
joblib.dump(pass_model,  os.path.join(MODELS_DIR,"pass_model.pkl"))
joblib.dump(grade_model, os.path.join(MODELS_DIR,"grade_model.pkl"))
joblib.dump(scaler,      os.path.join(MODELS_DIR,"scaler.pkl"))
joblib.dump(kmeans,      os.path.join(MODELS_DIR,"kmeans.pkl"))

print("\n✅ All models saved in:", MODELS_DIR)
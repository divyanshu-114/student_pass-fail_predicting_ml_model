# Architectural Data Pipeline
### Student Pass/Fail Prediction System — End-to-End Data Flow

---

```
┌─────────────────────────────────────────────────────────────┐
│                     📂 RAW INPUT DATA                       │
│           student_performance.csv  (1,000,000 rows)         │
│                                                             │
│  Columns: student_id, weekly_self_study_hours,              │
│           attendance_percentage, class_participation,       │
│           total_score, grade                                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  🧹 STEP 1: NULL HANDLING                   │
│                                                             │
│  df.fillna(df.median(numeric_only=True))                    │
│  → All missing values replaced with column median           │
│  → Result: 0 null values remain                             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               📊 STEP 2: OUTLIER REMOVAL (IQR)              │
│                                                             │
│  For each numeric column:                                   │
│    Lower Bound = Q1 − (1.5 × IQR)                           │
│    Upper Bound = Q3 + (1.5 × IQR)                           │
│  → Rows outside bounds are dropped                          │
│                                                             │
│  Before: 1,000,000 rows                                     │
│  After:  ~986,175 rows  (~13,825 removed)                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             🏷️  STEP 3: PASS/FAIL LABEL CREATION            │
│                                                             │
│  PASS if ALL three conditions are met:                      │
│    ✅ weekly_self_study_hours  ≥ 10                         │
│    ✅ attendance_percentage    ≥ 75                         │
│    ✅ class_participation      ≥ 5                          │
│  Otherwise → FAIL (0)                                       │
│                                                             │
│  Result: Pass = 448,136  |  Fail = 538,039                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            🔤 STEP 4: GRADE LABEL ENCODING                  │
│                                                             │
│  LabelEncoder().fit_transform(df["grade"])                  │
│    A → 0,  B → 1,  C → 2,  D → 3,  F → 4                    │
│  → Saved as: grade_encoder.pkl                              │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 🎯 STEP 5: FEATURE SELECTION                │
│                                                             │
│  Features used (X):                                         │
│    • weekly_self_study_hours                                │
│    • attendance_percentage                                  │
│    • class_participation                                    │
│                                                             │
│  ❌ Excluded (to prevent data leakage):                     │
│    • total_score,  grade,  student_id                       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                ⚖️  STEP 6: FEATURE SCALING                  │
│                                                             │
│  MinMaxScaler().fit_transform(X)                            │
│  → All values normalized to range [0.0 → 1.0]               │
│  → Saved as: scaler.pkl                                     │
│                                                             │
│  Ensures no single feature dominates due to scale           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               ✂️  STEP 7: TRAIN / TEST SPLIT                │
│                                                             │
│  train_test_split(X_scaled, y, test_size=0.2,               │
│                   random_state=42, stratify=y)              │
│                                                             │
│  Training Set:  ~788,940 rows  (80%)                        │
│  Test Set:      ~197,235 rows  (20%)                        │
│  Stratified → same Pass/Fail ratio in both splits           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              🔁 STEP 8: SMOTE OVERSAMPLING                  │
│                  (applied to TRAIN SET only)                │
│                                                             │
│  SMOTE(random_state=42).fit_resample(X_train, y_train)      │
│  → Generates synthetic minority-class samples               │
│  → Balances Fail class to match Pass count                  │
│  → Prevents model from always predicting "Pass"             │
└──────────────────────────────┬──────────────────────────────┘
                               │
                         ┌─────┴─────┐
                         ▼           ▼
        ┌────────────────────┐   ┌──────────────────────────┐
        │  🌲 MODEL A        │   │  🌲 MODEL B               │
        │  Pass/Fail         │   │  Grade Predictor          │
        │  Classifier        │   │  (A, B, C, D, F)          │
        │                    │   │                           │
        │  XGBoost           │   │  XGBoost                  │
        │  Classifier        │   │  Classifier               │
        │  (SMOTE train set) │   │  (full scaled data)       │
        │                    │   │                          │
        │ → pass_model.pkl   │   │ → grade_model.pkl        │
        └────────────────────┘   └──────────────────────────┘
                         │           │
                         └─────┬─────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│            🔵 STEP 9: K-MEANS CLUSTERING                    │
│                  (Unsupervised — no labels)                 │
│                                                             │
│  KMeans(n_clusters=3, random_state=42).fit(X_scaled)        │
│  → Segments all students into 3 behavioral clusters:        │
│      Cluster 0: Low engagement                              │
│      Cluster 1: Moderate engagement                         │
│      Cluster 2: High engagement                             │
│  → Saved as: kmeans.pkl                                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│               💾 STEP 10: MODEL SERIALIZATION                │
│                                                             │
│  All artifacts saved to /models/ directory:                 │
│                                                             │
│  ┌─────────────────────┬──────────────────────────────┐     │
│  │ File                │ Purpose                      │     │
│  ├─────────────────────┼──────────────────────────────┤     │
│  │ pass_model.pkl      │ Binary classifier (0/1)      │     │
│  │ grade_model.pkl     │ Grade predictor (A–F)        │     │
│  │ grade_encoder.pkl   │ Decode numeric → letter      │     │
│  │ scaler.pkl          │ Feature normalizer           │     │
│  │ kmeans.pkl          │ Behavioral cluster model     │     │
│  └─────────────────────┴──────────────────────────────┘     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           🖥️  STEP 11: STREAMLIT DASHBOARD (app.py)         │
│                                                             │
│  On user input:                                             │
│    1. User sets sliders → study hrs, attendance, part.      │
│    2. Input → DataFrame  →  scaler.transform()              │
│    3. pass_model.predict_proba() → Pass / Fail + %          │
│    4. grade_model.predict() → Letter grade (decoded)        │
│    5. kmeans.predict() → Behavioral cluster (0, 1, 2)       │
│    6. If Fail → show improvement suggestions                │
│                                                             │
│  Batch Mode:                                                │
│    Upload CSV → apply above pipeline to all rows            │
│    → Display results table + charts + download CSV          │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     📤 FINAL OUTPUT                         │
│                                                             │
│  Individual:  Pass ✅ / Fail ❌ + Probability %             │
│               Predicted Grade (A–F)                         │
│               Behavioral Cluster (0–2)                      │
│               Improvement Tips (if failing)                 │
│                                                             │
│  Batch:       Full results table per student                │
│               Grade distribution bar chart                  │
│               Pass vs Fail pie chart                        │
│               Cluster distribution bar chart                │
│               Downloadable results.csv                      │
└─────────────────────────────────────────────────────────────┘
```

---

**Team:** Divyanshu Raj · Yash Agarwal · Abhijeet · Ranajeet Roy

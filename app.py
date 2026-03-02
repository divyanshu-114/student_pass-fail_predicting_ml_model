import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# ─────────────────────────────
# LOAD MODELS
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

pass_model     = joblib.load(os.path.join(MODELS_DIR,"pass_model.pkl"))
grade_model    = joblib.load(os.path.join(MODELS_DIR,"grade_model.pkl"))
grade_encoder  = joblib.load(os.path.join(MODELS_DIR,"grade_encoder.pkl"))
scaler         = joblib.load(os.path.join(MODELS_DIR,"scaler.pkl"))
kmeans         = joblib.load(os.path.join(MODELS_DIR,"kmeans.pkl"))

FEATURES = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation"
]

# ─────────────────────────────
# IMPROVEMENT SUGGESTIONS
# ─────────────────────────────
def improvement_areas(row):
    tips = []

    if row["weekly_self_study_hours"] < 10:
        tips.append("Increase study hours")

    if row["attendance_percentage"] < 75:
        tips.append("Improve attendance")

    if row["class_participation"] < 5:
        tips.append("Participate more in class")

    return tips if tips else ["Keep up the good work!"]

# ─────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────
def predict(df):

    X = df[FEATURES]
    X_scaled = scaler.transform(X)

    prob = pass_model.predict_proba(X_scaled)[:,1]
    pred = (prob >= 0.5).astype(int)

    grade_nums = grade_model.predict(X_scaled)
    grades = grade_encoder.inverse_transform(grade_nums)

    cluster = kmeans.predict(X_scaled)

    return prob, pred, grades, cluster


# ─────────────────────────────
# SIDEBAR
# ─────────────────────────────
st.sidebar.title("Dashboard")
page = st.sidebar.radio("Select", ["Individual Prediction","Batch Prediction"])

# ═════════════════════════════
# 🎓 INDIVIDUAL
# ═════════════════════════════
if page == "Individual Prediction":

    st.title("🎓 Student Individual Prediction")

    study = st.slider("Study Hours",0,40,10)
    attend = st.slider("Attendance %",0,100,80)
    part = st.slider("Participation",0,10,5)

    if st.button("Predict"):

        df = pd.DataFrame([{
            "weekly_self_study_hours":study,
            "attendance_percentage":attend,
            "class_participation":part
        }])

        prob, pred, grades, cluster = predict(df)

        st.subheader("Result")

        if pred[0] == 1:
            st.success("Pass ✅")
        else:
            st.error("Fail ❌")
            st.subheader("Area of Improvement")
            for tip in improvement_areas(df.iloc[0]):
                st.write("•", tip)

        st.write("Probability:", round(prob[0]*100,2),"%")
        st.write("Grade:", grades[0])
        st.write("Cluster:", cluster[0])


# ═════════════════════════════
# 📁 BATCH
# ═════════════════════════════
else:

    st.title("📁 CSV Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        df = pd.read_csv(file)

        missing = [c for c in FEATURES if c not in df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        prob, pred, grades, cluster = predict(df)

        df["Pass_Fail"] = pred
        df["Probability"] = prob
        df["Grade"] = grades
        df["Cluster"] = cluster

        # Improvement for failed students
        df["Improvement"] = df.apply(
            lambda row: ", ".join(improvement_areas(row)) if row["Pass_Fail"]==0 else "—",
            axis=1
        )

        st.dataframe(df)

        # 🔎 SEARCH STUDENT
        st.subheader("🔎 Search by Student ID")

        if "student_id" in df.columns:

            sid = st.text_input("Enter Student ID")

            if st.button("Search"):

                result = df[df["student_id"].astype(str)==sid]

                if not result.empty:
                    st.dataframe(result)
                else:
                    st.warning("Student not found")

        # 📊 CHARTS
        st.subheader("📊 Charts")

        col1, col2 = st.columns(2)

        # Grade chart
        with col1:
            fig, ax = plt.subplots()
            df["Grade"].value_counts().plot(kind="bar", ax=ax)
            ax.set_title("Grade Distribution")
            st.pyplot(fig)

        # Pass Fail chart
        with col2:
            fig, ax = plt.subplots()
            df["Pass_Fail"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_title("Pass vs Fail")
            st.pyplot(fig)

        # Cluster chart
        fig, ax = plt.subplots()
        df["Cluster"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Cluster Distribution")
        st.pyplot(fig)

        # DOWNLOAD
        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "results.csv"
        )
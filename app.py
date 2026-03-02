"""Streamlit dashboard for visualizing predictions."""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from src.inference import StudentPredictor
from src.ui_components import UIBuilder
from src.config import FEATURES

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# Initialize models and UI tools
ui = UIBuilder()
ui.load_css()
predictor = StudentPredictor()

if not predictor.is_ready():
    st.error("⚠️ Models not found! Please run `python train_model.py` first.")
    st.stop()

# Dashboard Switcher
st.sidebar.title("Dashboard")
page = st.sidebar.radio("Select View", ["Individual Prediction", "Batch Prediction"])

if page == "Individual Prediction":
    st.title("🎓 Student Individual Prediction")

    col1, col2, col3 = st.columns(3)
    study = col1.slider("Weekly Study Hours", 0, 40, 10)
    attend = col2.slider("Attendance %", 0, 100, 80)
    part = col3.slider("Participation", 0, 10, 5)

    if st.button("Predict Outcome", type="primary"):
        df = pd.DataFrame([{
            "weekly_self_study_hours": study,
            "attendance_percentage": attend,
            "class_participation": part
        }])

        prob, pred, cluster = predictor.predict_bundle(df)
        
        # Render clean result
        ui.render_prediction_card(prob[0], pred[0], cluster[0])
        
        # Area of Improvement
        if pred[0] == 0:
            st.subheader("Actionable Recommendations")
            for tip in predictor.get_student_recommendations(df.iloc[0]):
                st.write(f"• {tip}")

else:
    st.title("📁 CSV Batch Prediction")
    file = st.file_uploader("Upload Student Data (CSV)", type=["csv"])

    if file:
        df = pd.read_csv(file)
        missing = [c for c in FEATURES if c not in df.columns]

        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        # Run batch prediction
        prob, pred, cluster = predictor.predict_bundle(df)
        
        df["Pass_Fail"] = pred
        df["Probability"] = prob
        df["Cluster"] = cluster

        # Improvement tips
        df["Improvement"] = df.apply(
            lambda row: ", ".join(predictor.get_student_recommendations(row)) if row["Pass_Fail"] == 0 else "—",
            axis=1
        )

        st.dataframe(df)

        st.subheader("🔎 Search by Student ID")
        if "student_id" in df.columns:
            sid = st.text_input("Enter Student ID:")
            if st.button("Search"):
                result = df[df["student_id"].astype(str) == str(sid)]
                if not result.empty:
                    st.dataframe(result)
                else:
                    st.warning("Student ID not found in the uploaded file.")

        st.subheader("📊 Visualizations")
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            pf_counts = df["Pass_Fail"].map({1: "Pass", 0: "Fail"}).value_counts()
            ui.mpl_pie(ax, pf_counts.index.tolist(), pf_counts.values.tolist(), "Pass vs Fail Ratio")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            c_counts = df["Cluster"].value_counts().sort_index()
            ui.mpl_bar(ax, [f"C{x}" for x in c_counts.index], c_counts.values.tolist(), "Cluster Distribution", ["#a78bfa"]*len(c_counts))
            st.pyplot(fig)

        st.download_button(
            "Download Batch Results",
            df.to_csv(index=False),
            "predicted_results.csv"
        )

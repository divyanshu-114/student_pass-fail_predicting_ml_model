import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

# -----------------------------
# PDF table extraction (optional)
# -----------------------------
CAMELOT_AVAILABLE = False
try:
    import camelot  # pip install camelot-py[cv] pypdf
    CAMELOT_AVAILABLE = True
except Exception:
    CAMELOT_AVAILABLE = False


# =============================
# 🎨 PAGE CONFIG + MINIMAL CSS
# =============================
st.set_page_config(
    page_title="Intelligent Learning Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
hr { opacity: 0.15; }
.small-note { font-size: 0.9rem; opacity: 0.7; }
.card {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 14px;
    background: rgba(255,255,255,0.02);
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("📊 Intelligent Learning Analytics Dashboard")
st.caption("Minimal, clean analytics + predictions for student performance.")

FEATURE_COLS = ["weekly_self_study_hours", "attendance_percentage", "class_participation"]


# =============================
# 🔧 HELPERS
# =============================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def make_pass_column(df: pd.DataFrame) -> pd.DataFrame:
    if "pass" in df.columns:
        df["pass"] = df["pass"].apply(
            lambda x: 1 if str(x).strip() in ["1", "True", "true", "PASS", "pass"] else 0
        )
        return df

    if "grade" in df.columns:
        df["pass"] = df["grade"].astype(str).str.upper().apply(
            lambda g: 1 if g in ["A", "B", "C"] else 0
        )
        return df

    if "total_score" in df.columns:
        df["pass"] = (
            pd.to_numeric(df["total_score"], errors="coerce").fillna(0) >= 50
        ).astype(int)
        return df

    df["pass"] = np.nan
    return df


def validate_dataset(df: pd.DataFrame) -> list:
    issues = []
    for col in FEATURE_COLS:
        if col not in df.columns:
            issues.append(f"Missing required column: `{col}`")
    if "pass" not in df.columns or df["pass"].isna().all():
        issues.append("Could not infer `pass` labels. Provide `pass` or `grade` (or `total_score`).")
    return issues


@st.cache_data(show_spinner=False)
def load_dataset_from_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    return normalize_columns(df)


@st.cache_data(show_spinner=False)
def load_dataset_from_pdf(file) -> pd.DataFrame:
    if not CAMELOT_AVAILABLE:
        raise RuntimeError("Camelot not installed. Install: pip install camelot-py[cv] pypdf")

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getbuffer())
        tmp_path = tmp.name

    try:
        tables = camelot.read_pdf(tmp_path, pages="all", flavor="lattice")
    except Exception:
        tables = camelot.read_pdf(tmp_path, pages="all", flavor="stream")

    if tables is None or tables.n == 0:
        raise RuntimeError("No tables found in the PDF.")

    best = max(tables, key=lambda t: t.df.shape[0] * t.df.shape[1]).df
    best.columns = best.iloc[0]
    best = best.iloc[1:].reset_index(drop=True)
    return normalize_columns(best)


def student_recommendations(row: pd.Series, cohort: pd.DataFrame) -> list:
    recs = []
    med = cohort[FEATURE_COLS].median(numeric_only=True)

    if row["weekly_self_study_hours"] < 15:
        recs.append("Increase weekly self-study to **15–20 hours** with a consistent schedule.")
    if row["attendance_percentage"] < 80:
        recs.append("Aim for **80%+ attendance** to reduce missed concepts.")
    if row["class_participation"] < 5:
        recs.append("Improve participation: ask questions + attempt in-class problems.")

    if row["weekly_self_study_hours"] < med["weekly_self_study_hours"]:
        recs.append("Study hours are **below cohort median** — add 30–60 minutes daily.")
    if row["attendance_percentage"] < med["attendance_percentage"]:
        recs.append("Attendance is **below cohort median** — set a weekly attendance goal.")
    if row["class_participation"] < med["class_participation"]:
        recs.append("Participation is **below cohort median** — prepare 1–2 questions per class.")

    if not recs:
        recs.append("Maintain consistency. Add spaced revision + advanced practice sets weekly.")
    return recs[:5]


@st.cache_resource(show_spinner=False)
def fit_pipeline(df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    clean = df.copy()

    for col in FEATURE_COLS:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=FEATURE_COLS + ["pass"]).copy()
    clean["pass"] = clean["pass"].astype(int)

    X = clean[FEATURE_COLS]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(Xs, clean["pass"])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    kmeans.fit(Xs)

    scored = clean.copy()
    scored["cluster"] = kmeans.predict(Xs)
    scored["pred_proba"] = clf.predict_proba(Xs)[:, 1]
    scored["prediction"] = (scored["pred_proba"] >= 0.5).astype(int)

    cm = confusion_matrix(scored["pass"], scored["prediction"], labels=[0, 1])
    return {"scaler": scaler, "clf": clf, "kmeans": kmeans, "scored_df": scored, "cm": cm}


# =============================
# 📊 MATPLOTLIB CHART HELPERS
# =============================
def mpl_hist(ax, data, title, xlabel):
    ax.hist(data.dropna(), bins=20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")


def mpl_bar_counts(ax, labels, values, title, xlabel, ylabel):
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def mpl_pie(ax, labels, values, title):
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)


def mpl_heatmap(ax, mat, title, xlabels, ylabels):
    im = ax.imshow(mat, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def mpl_box_by_cluster(ax, df, col, title):
    groups = []
    labels = []
    for c in sorted(df["cluster"].unique().tolist()):
        groups.append(pd.to_numeric(df[df["cluster"] == c][col], errors="coerce").dropna().values)
        labels.append(str(c))

    ax.boxplot(groups, labels=labels)
    ax.set_title(title)
    ax.set_xlabel("Cluster")
    ax.set_ylabel(col)


# =============================
# 🧠 SESSION STATE (shared model)
# =============================
if "shared_model" not in st.session_state:
    st.session_state.shared_model = None
if "shared_scored" not in st.session_state:
    st.session_state.shared_scored = None
if "fingerprint" not in st.session_state:
    st.session_state.fingerprint = None


# =============================
# 🧭 SIDEBAR NAV
# =============================
st.sidebar.header("Navigation")
mode = st.sidebar.radio("Choose a section", ["Upload PDF/CSV Dataset", "Individual Student Prediction"])


# =============================
# 📄 MODE 1: UPLOAD DATASET
# =============================
if mode == "Upload PDF/CSV Dataset":
    st.subheader("Upload Dataset (PDF or CSV)")

    colA, colB = st.columns([2, 1], gap="large")
    with colA:
        uploaded = st.file_uploader("Upload a PDF table or CSV file", type=["pdf", "csv"])
        n_clusters = st.slider("Number of clusters", 2, 6, 3)
    with colB:
        st.markdown(
            """
<div class="card">
<b>Tip</b><br><br>
If PDF parsing fails, export/convert it to CSV for best reliability.
</div>
""",
            unsafe_allow_html=True,
        )

    if uploaded is None:
        st.stop()

    # Load dataset
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_raw = load_dataset_from_csv(uploaded)
        else:
            df_raw = load_dataset_from_pdf(uploaded)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        if uploaded.name.lower().endswith(".pdf") and not CAMELOT_AVAILABLE:
            st.info("To enable PDF parsing: `pip install camelot-py[cv] pypdf`")
        st.stop()

    df_raw = make_pass_column(df_raw)
    issues = validate_dataset(df_raw)
    if issues:
        st.error("Dataset validation failed:")
        for it in issues:
            st.write(f"• {it}")
        st.stop()

    st.markdown("### Dataset Preview")
    st.dataframe(df_raw.head(25), use_container_width=True)

    # Auto-train if file/cluster changed
    fp = f"{uploaded.name}-{uploaded.size}-k{n_clusters}"
    if st.session_state.fingerprint != fp:
        with st.spinner("Training model from uploaded dataset..."):
            bundle = fit_pipeline(df_raw, n_clusters=n_clusters)
            st.session_state.shared_model = bundle
            st.session_state.shared_scored = bundle["scored_df"]
            st.session_state.fingerprint = fp

    scored = st.session_state.shared_scored
    cm = st.session_state.shared_model["cm"]

    st.markdown("---")
    tab1, tab2 = st.tabs(["Student Search", "Insights & Charts"])

    # -----------------------------
    # Student Search (ONLY typed)
    # -----------------------------
    with tab1:
        st.markdown("### Search a Student")

        if "student_id" not in scored.columns:
            st.info("`student_id` column not found in dataset.")
        else:
            typed_id = st.text_input("Type student_id", value="")

            if typed_id.strip():
                chosen = typed_id.strip()
                match = scored[scored["student_id"].astype(str) == chosen]

                if match.empty:
                    st.error("Student ID not found.")
                else:
                    srow = match.iloc[0]
                    left, right = st.columns([1.2, 1], gap="large")

                    with left:
                        pred_label = "PASS ✅" if int(srow["prediction"]) == 1 else "FAIL ❌"
                        st.markdown(
                            f"""
<div class="card">
<b>student_id:</b> {chosen}<br>
<b>Predicted:</b> {pred_label}<br>
<b>Pass probability:</b> {float(srow["pred_proba"])*100:.1f}%<br>
<b>Cluster:</b> {int(srow["cluster"])}<br>
</div>
""",
                            unsafe_allow_html=True,
                        )

                        st.markdown("#### Inputs")
                        st.write(pd.DataFrame({
                            "weekly_self_study_hours": [srow["weekly_self_study_hours"]],
                            "attendance_percentage": [srow["attendance_percentage"]],
                            "class_participation": [srow["class_participation"]],
                        }))

                    with right:
                        st.markdown("#### Suggestions")
                        recs = student_recommendations(srow, scored)
                        if int(srow["prediction"]) == 1:
                            st.success("Predicted PASS — keep consistency and revise weekly.")
                        else:
                            st.warning("Predicted FAIL — focus on these actions:")
                        for r in recs:
                            st.write(f"• {r}")
            else:
                st.info("Type a student_id above to search.")

    # -----------------------------
    # Insights & Charts (Matplotlib)
    # -----------------------------
    with tab2:
        st.markdown("### Insights & Charts")

        # Row 1: Histograms
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        mpl_hist(axes[0], pd.to_numeric(scored["weekly_self_study_hours"], errors="coerce"),
                 "Study Hours Distribution", "weekly_self_study_hours")
        mpl_hist(axes[1], pd.to_numeric(scored["attendance_percentage"], errors="coerce"),
                 "Attendance Distribution", "attendance_percentage")
        mpl_hist(axes[2], pd.to_numeric(scored["class_participation"], errors="coerce"),
                 "Participation Distribution", "class_participation")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

        # Row 2: Pass/Fail bar + pie
        pred_counts = scored["prediction"].value_counts().reindex([1, 0]).fillna(0).astype(int)
        pass_count = int(pred_counts.get(1, 0))
        fail_count = int(pred_counts.get(0, 0))

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        mpl_bar_counts(
            axes2[0],
            labels=["Pass", "Fail"],
            values=[pass_count, fail_count],
            title="Predicted Pass vs Fail (Bar)",
            xlabel="Prediction",
            ylabel="Count",
        )
        mpl_pie(
            axes2[1],
            labels=["Pass", "Fail"],
            values=[pass_count, fail_count],
            title="Predicted Pass vs Fail (Pie)",
        )
        plt.tight_layout()
        st.pyplot(fig2)

        st.markdown("---")

        # Row 3: Cluster distribution + confusion matrix
        cluster_counts = scored["cluster"].value_counts().sort_index()
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
        mpl_bar_counts(
            axes3[0],
            labels=[str(i) for i in cluster_counts.index.tolist()],
            values=cluster_counts.values.tolist(),
            title="Cluster Distribution",
            xlabel="Cluster",
            ylabel="Count",
        )
        mpl_heatmap(
            axes3[1],
            cm,
            "Confusion Matrix",
            xlabels=["Pred Fail", "Pred Pass"],
            ylabels=["Actual Fail", "Actual Pass"],
        )
        plt.tight_layout()
        st.pyplot(fig3)

        st.markdown("---")

        # Row 4: Boxplots by cluster
        if "cluster" in scored.columns and scored["cluster"].nunique() >= 2:
            fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))
            mpl_box_by_cluster(axes4[0], scored, "weekly_self_study_hours", "Study Hours by Cluster")
            mpl_box_by_cluster(axes4[1], scored, "attendance_percentage", "Attendance by Cluster")
            mpl_box_by_cluster(axes4[2], scored, "class_participation", "Participation by Cluster")
            plt.tight_layout()
            st.pyplot(fig4)

        st.markdown("---")

        # Row 5: Correlation heatmap
        corr_df = scored.copy()
        for c in FEATURE_COLS:
            corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")
        corr = corr_df[FEATURE_COLS].corr().values

        fig5, ax5 = plt.subplots(figsize=(6, 4))
        mpl_heatmap(
            ax5,
            corr,
            "Correlation Heatmap (Features)",
            xlabels=FEATURE_COLS,
            ylabels=FEATURE_COLS,
        )
        plt.tight_layout()
        st.pyplot(fig5)


# =============================
# 🎯 MODE 2: INDIVIDUAL PREDICTION (uses uploaded model)
# =============================
else:
    st.subheader("Individual Student Prediction")

    if st.session_state.shared_model is None:
        st.warning("Upload a dataset in the first section to enable predictions.")
        st.stop()

    bundle = st.session_state.shared_model
    scaler = bundle["scaler"]
    clf = bundle["clf"]
    kmeans = bundle["kmeans"]
    scored_ref = st.session_state.shared_scored

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### Enter Student Details")
        study_hours = st.slider("Weekly Self Study Hours", 0, 40, 10)
        attendance = st.slider("Attendance Percentage", 0, 100, 80)
        participation = st.slider("Class Participation (0–10)", 0, 10, 5)
        predict_btn = st.button("Predict", type="primary")

    with right:
        if predict_btn:
            inp = pd.DataFrame([{
                "weekly_self_study_hours": study_hours,
                "attendance_percentage": attendance,
                "class_participation": participation,
            }])

            X_in = scaler.transform(inp[FEATURE_COLS])
            proba = clf.predict_proba(X_in)[0, 1]
            pred = 1 if proba >= 0.5 else 0
            cluster = int(kmeans.predict(X_in)[0])

            st.markdown("### Output")

            o1, o2, o3 = st.columns(3, gap="large")
            o1.metric("Pass probability", f"{proba * 100:.1f}%")
            o2.metric("Prediction", "PASS ✅" if pred == 1 else "FAIL ❌")
            o3.metric("Cluster", str(cluster))

            pseudo = pd.Series({
                "weekly_self_study_hours": study_hours,
                "attendance_percentage": attendance,
                "class_participation": participation,
            })
            recs = student_recommendations(pseudo, scored_ref)

            st.markdown("#### Suggestions")
            for r in recs:
                st.write(f"• {r}")

            # Small clean chart: position vs dataset (matplotlib scatter)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(
                pd.to_numeric(scored_ref["weekly_self_study_hours"], errors="coerce"),
                pd.to_numeric(scored_ref["attendance_percentage"], errors="coerce"),
                alpha=0.25,
                label="Dataset",
            )
            ax.scatter([study_hours], [attendance], marker="X", s=120, label="You")
            ax.set_title("Your Position vs Dataset")
            ax.set_xlabel("weekly_self_study_hours")
            ax.set_ylabel("attendance_percentage")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.empty()
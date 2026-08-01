"""
app.py -- AI Diagnostic Assistant for Oncology Screening
Streamlit dashboard built on top of train_model.py's saved artifact.

Run locally with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
import io

st.set_page_config(page_title="AI Diagnostic Assistant - Oncology Screening", layout="wide")

# ---------------------------------------------------------------------------
# LOAD MODEL + EXPLAINER (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifact():
    with open("diagnostic_model.pkl", "rb") as f:
        return pickle.load(f)

artifact = load_artifact()
model = artifact["model"]
explainer = artifact["explainer"]
feature_names = artifact["feature_names"]
test_metrics = artifact["test_metrics"]

# Session-level "screening log" -- simulates a running record of screened patients.
# NOTE: this resets when the app restarts. For a persistent version across real
# sessions/users, swap this for a small SQLite table (a few lines with sqlite3).
if "screening_log" not in st.session_state:
    st.session_state.screening_log = []

# ---------------------------------------------------------------------------
# RISK STRATIFICATION LOGIC
# ---------------------------------------------------------------------------
def risk_band(probability_malignant: float) -> tuple[str, str]:
    """
    Converts a raw probability into a clinical-style risk band.
    Thresholds below are a reasonable starting point for a screening tool --
    in a real clinical setting these would be set with a domain expert,
    often favoring high recall (catching more at the cost of more false alarms).
    """
    if probability_malignant < 0.30:
        return "Low Risk", "green"
    elif probability_malignant < 0.65:
        return "Medium Risk", "orange"
    else:
        return "High Risk", "red"


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Navigate", ["Single Patient Screening", "Batch Upload", "Screening Analytics", "Model Performance"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "This tool is a decision-support demo built on the Wisconsin Breast Cancer "
    "diagnostic dataset. It is NOT a certified medical device and should never "
    "be used for real clinical decisions."
)

# ===========================================================================
# PAGE 1: SINGLE PATIENT SCREENING
# ===========================================================================
if page == "Single Patient Screening":
    st.title("🩺 AI Diagnostic Assistant — Oncology Screening")
    st.write("Enter lab feature values for a patient to get a risk assessment.")

    with st.expander("ℹ️ How to use this", expanded=False):
        st.write(
            "Enter the 30 diagnostic measurements from a breast mass biopsy "
            "(radius, texture, perimeter, area, etc. -- mean, standard error, "
            "and worst-case values). These are the same features used in the "
            "original Wisconsin Diagnostic Breast Cancer dataset."
        )

    # Group features into mean / se / worst for a cleaner form layout.
    # NOTE: sklearn's built-in dataset names columns like "mean radius",
    # "radius error", "worst radius" -- prefix/suffix pattern, not
    # "radius_mean" style. This grouping matches THAT naming exactly.
    mean_feats = [f for f in feature_names if f.startswith("mean ")]
    se_feats = [f for f in feature_names if f.endswith(" error")]
    worst_feats = [f for f in feature_names if f.startswith("worst ")]

    input_vals = {}
    tab1, tab2, tab3 = st.tabs(["Mean Values", "Standard Error", "Worst Values"])

    def render_inputs(container, feats):
        cols = container.columns(3)
        for i, feat in enumerate(feats):
            with cols[i % 3]:
                input_vals[feat] = st.number_input(feat, min_value=0.0, value=1.0, step=0.1, key=feat)

    with tab1:
        render_inputs(tab1, mean_feats)
    with tab2:
        render_inputs(tab2, se_feats)
    with tab3:
        render_inputs(tab3, worst_feats)

    patient_id = st.text_input("Patient ID / Reference (optional)", value=f"P-{len(st.session_state.screening_log)+1:04d}")

    if st.button("Run Screening", type="primary"):
        X_input = pd.DataFrame([input_vals])[feature_names]  # preserve column order
        proba_malignant = model.predict_proba(X_input)[0][1]
        prediction = "Malignant" if proba_malignant >= 0.5 else "Benign"
        band, color = risk_band(proba_malignant)

        # --- Results display ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", prediction)
        col2.metric("Malignancy Probability", f"{proba_malignant*100:.1f}%")
        col3.markdown(f"### Risk Band: :{color}[{band}]")

        st.progress(min(max(proba_malignant, 0.0), 1.0))

        # --- SHAP explanation ---
        st.subheader("Why did the model decide this?")
        shap_values = explainer.shap_values(X_input)
        # For binary classifiers, shap_values can be a list [class0, class1] or a single array
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        shap_df = pd.DataFrame({"feature": feature_names, "impact": sv})
        shap_df["abs_impact"] = shap_df["impact"].abs()
        top_features = shap_df.sort_values("abs_impact", ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(7, 4))
        colors_bar = ["#d62728" if v > 0 else "#2ca02c" for v in top_features["impact"]]
        ax.barh(top_features["feature"], top_features["impact"], color=colors_bar)
        ax.set_xlabel("Impact on malignancy prediction (SHAP value)")
        ax.invert_yaxis()
        st.pyplot(fig)

        top_driver = top_features.iloc[0]
        direction = "toward Malignant" if top_driver["impact"] > 0 else "toward Benign"
        st.info(
            f"**Interpretation:** The model was pushed most strongly {direction} by "
            f"**{top_driver['feature']}**. Red bars push toward Malignant, green bars push toward Benign."
        )

        # --- Log this screening for the analytics page ---
        st.session_state.screening_log.append({
            "patient_id": patient_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "prediction": prediction,
            "probability": proba_malignant,
            "risk_band": band,
        })

        # --- PDF report generation ---
        def build_pdf():
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Oncology Screening Report", ln=True)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 8, f"Patient: {patient_id}", ln=True)
            pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 8, f"Prediction: {prediction}", ln=True)
            pdf.cell(0, 8, f"Malignancy Probability: {proba_malignant*100:.1f}%", ln=True)
            pdf.cell(0, 8, f"Risk Band: {band}", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Top contributing features:", ln=True)
            pdf.set_font("Helvetica", "", 11)
            for _, row in top_features.iterrows():
                sign = "+" if row["impact"] > 0 else "-"
                pdf.cell(0, 7, f"  {row['feature']}: {sign}{abs(row['impact']):.3f}", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 6, "Disclaimer: This is an AI decision-support demo, not a certified "
                                 "diagnostic device. All results must be reviewed by a qualified clinician.")
            return bytes(pdf.output(dest="S"))

        pdf_bytes = build_pdf()
        st.download_button("📄 Download PDF Report", data=pdf_bytes,
                            file_name=f"{patient_id}_screening_report.pdf", mime="application/pdf")

# ===========================================================================
# PAGE 2: BATCH UPLOAD
# ===========================================================================
elif page == "Batch Upload":
    st.title("📁 Batch Screening Upload")
    st.write("Upload a CSV with multiple patient records (same 30 feature columns) to screen them all at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        missing = set(feature_names) - set(batch_df.columns)
        if missing:
            st.error(f"CSV is missing required columns: {missing}")
        else:
            X_batch = batch_df[feature_names]
            probs = model.predict_proba(X_batch)[:, 1]
            preds = np.where(probs >= 0.5, "Malignant", "Benign")
            bands = [risk_band(p)[0] for p in probs]

            results = batch_df.copy()
            results["malignancy_probability"] = probs.round(3)
            results["prediction"] = preds
            results["risk_band"] = bands

            st.dataframe(results[["prediction", "risk_band", "malignancy_probability"]])

            for _, row in results.iterrows():
                st.session_state.screening_log.append({
                    "patient_id": f"batch-{len(st.session_state.screening_log)+1}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "prediction": row["prediction"],
                    "probability": row["malignancy_probability"],
                    "risk_band": row["risk_band"],
                })

            csv_out = results.to_csv(index=False).encode()
            st.download_button("⬇ Download Results CSV", data=csv_out, file_name="batch_screening_results.csv")

# ===========================================================================
# PAGE 3: SCREENING ANALYTICS (the "monitoring dashboard" feature)
# ===========================================================================
elif page == "Screening Analytics":
    st.title("📊 Screening Program Analytics")

    if not st.session_state.screening_log:
        st.info("No screenings run yet this session. Run some from the Single Patient or Batch pages first.")
    else:
        log_df = pd.DataFrame(st.session_state.screening_log)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Screened", len(log_df))
        col2.metric("Flagged High Risk", (log_df["risk_band"] == "High Risk").sum())
        col3.metric("Avg. Malignancy Probability", f"{log_df['probability'].mean()*100:.1f}%")

        st.subheader("Risk Band Distribution")
        st.bar_chart(log_df["risk_band"].value_counts())

        st.subheader("Screening Log")
        st.dataframe(log_df.sort_values("timestamp", ascending=False))

# ===========================================================================
# PAGE 4: MODEL PERFORMANCE (transparency page -- shows this isn't a black box)
# ===========================================================================
elif page == "Model Performance":
    st.title("📈 Model Performance & Validation")
    st.write("Performance of the underlying Decision Tree classifier, evaluated on a held-out test set.")

    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{test_metrics['accuracy']*100:.1f}%")
    cols[1].metric("Precision", f"{test_metrics['precision']*100:.1f}%")
    cols[2].metric("Recall", f"{test_metrics['recall']*100:.1f}%")
    cols[3].metric("F1 Score", f"{test_metrics['f1']*100:.1f}%")
    cols[4].metric("ROC-AUC", f"{test_metrics['roc_auc']*100:.1f}%")

    st.markdown(
        "**Why Recall matters most here:** Recall measures how many of the "
        "actual malignant cases the model correctly caught. In a screening "
        "context, missing a malignant case (a false negative) is far more "
        "costly than a false alarm, so this model was tuned to optimize recall "
        "during hyperparameter search, not just raw accuracy."
    )

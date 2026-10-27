"""Optional Streamlit demo for an educational model pipeline."""
from pathlib import Path

import pandas as pd
import streamlit as st
from joblib import load

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "heart_disease_pipeline.joblib"
DATA_PATH = ROOT / "data" / "heart.csv"

st.set_page_config(page_title="Heart Disease Prediction Demo", page_icon="🫀", layout="centered")
st.title("🫀 Heart Disease Prediction — Demo")
st.warning(
    "Educational demonstration only. This model is not clinically validated and must not be used "
    "for diagnosis, screening, treatment, or other medical decisions."
)
st.write("Enter example feature values to demonstrate how a trained classifier accepts structured input.")

if not MODEL_PATH.exists():
    st.info("Model file not found. From the project folder, run `python train.py` first.")
    st.stop()

reference = pd.read_csv(DATA_PATH).drop(columns=["target"])
model = load(MODEL_PATH)

with st.form("prediction_form"):
    values = {}
    cols = st.columns(2)
    for i, col in enumerate(reference.columns):
        series = reference[col]
        if col in {"sex", "fbs", "exang", "cp", "restecg", "slope", "ca", "thal"}:
            options = sorted(series.dropna().unique().tolist())
            values[col] = cols[i % 2].selectbox(col, options=options, help="Dataset-coded category; see dataset documentation.")
        else:
            values[col] = cols[i % 2].number_input(
                col,
                min_value=float(series.min()),
                max_value=float(series.max()),
                value=float(series.median()),
                help="Input range is limited to values observed in this dataset for demonstration.",
            )
    submitted = st.form_submit_button("Run demo prediction")

if submitted:
    sample = pd.DataFrame([values], columns=reference.columns)
    prediction = int(model.predict(sample)[0])
    st.subheader("Model output")
    st.write(f"Predicted dataset class: **{prediction}**")
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(sample)[0, 1])
        st.write(f"Model-estimated probability for class 1: **{probability:.1%}**")
    st.caption("This is a model output on an educational dataset, not an estimate of an individual's medical risk.")

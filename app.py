# app.py
import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Load the trained model (once)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("cvd_model_perfect.pkl")

model = load_model()

st.title("CVD 10-Year Risk Predictor")
st.write(
    "Upload a CSV with the same columns as `data_cvd_perfect.csv` "
    "(note, age, sys_bp, dia_bp, cholesterol, glucose, bmi, smoke, family_hx). "
    "The app will return the predicted risk percentage."
)

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # Predict
    prob = model.predict_proba(df)[:, 1]
    df["CVD_Risk_%"] = (prob * 100).round(1)

    st.write("### Predictions")
    st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False).encode()
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name="cvd_risk_results.csv",
        mime="text/csv",
    )
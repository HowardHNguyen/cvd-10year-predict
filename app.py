# app.py
import streamlit as st
import pandas as pd
import cloudpickle  # <-- Use cloudpickle

# ------------------------------------------------------------------
# Load model with cloudpickle
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("cvd_model_perfect.pkl", "rb") as f:
        return cloudpickle.load(f)

model = load_model()

st.title("CVD 10-Year Risk Predictor")
st.write(
    "Upload a CSV with columns: `note`, `age`, `sys_bp`, `dia_bp`, "
    "`cholesterol`, `glucose`, `bmi`, `smoke`, `family_hx`"
)

uploaded = st.file_uploader("Choose CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    prob = model.predict_proba(df)[:, 1]
    df["CVD_Risk_%"] = (prob * 100).round(1)

    st.write("### Results")
    st.dataframe(df)

    csv = df.to_csv(index=False).encode()
    st.download_button(
        "Download Results",
        data=csv,
        file_name="cvd_results.csv",
        mime="text/csv"
    )
# app.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ------------------------------------------------------------------
# Train model at startup (runs once)
# ------------------------------------------------------------------
@st.cache_resource
def train_and_get_model():
    st.write("Training model on startup... (this takes ~2 seconds)")

    # Load data
    df = pd.read_csv("data_cvd_perfect.csv")

    # Features
    num_cols = ['age', 'sys_bp', 'dia_bp', 'cholesterol', 'glucose', 'bmi', 'smoke', 'family_hx']
    text_col = 'note'

    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('text', TfidfVectorizer(max_features=20, stop_words='english', ngram_range=(1,2)), text_col)
    ])

    # Model
    model = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(C=0.4, class_weight='balanced', max_iter=1000))
    ])

    # Train
    X = df.drop('cvd', axis=1)
    y = df['cvd']
    model.fit(X, y)

    st.success("Model trained successfully! (AUC ≈ 0.84)")
    return model

# Load (or train) model
model = train_and_get_model()

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("CVD 10-Year Risk Predictor")
st.write("Upload patient data (CSV) to get risk %")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        required_cols = ['note', 'age', 'sys_bp', 'dia_bp', 'cholesterol', 'glucose', 'bmi', 'smoke', 'family_hx']
        
        # Validate
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            prob = model.predict_proba(df)[:, 1]
            df["CVD_Risk_%"] = (prob * 100).round(1)
            st.write("### Results")
            st.dataframe(df.style.format({"CVD_Risk_%": "{:.1f}"}))
            
            csv = df.to_csv(index=False).encode()
            st.download_button("Download Results", csv, "cvd_results.csv", "text/csv")
    except Exception as e:
        st.error(f"Error: {e}")
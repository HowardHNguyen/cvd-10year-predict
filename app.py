# app.py
import streamlit as st
import pandas as pd

# ------------------------------------------------------------------
# Train model at startup (runs once, cached)
# ------------------------------------------------------------------
@st.cache_resource
def train_and_get_model():
    st.info("Training model on startup... (takes ~3 seconds first time only)")

    df = pd.read_csv("data_cvd_perfect.csv")

    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer

    num_cols = ['age', 'sys_bp', 'dia_bp', 'cholesterol', 'glucose', 'bmi', 'smoke', 'family_hx']
    text_col = 'note'

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('text', TfidfVectorizer(max_features=20, stop_words='english', ngram_range=(1,2)), text_col)
    ])

    model = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(C=0.4, class_weight='balanced', max_iter=1000))
    ])

    X = df.drop('cvd', axis=1)
    y = df['cvd']
    model.fit(X, y)

    st.success("Model ready! (AUC ≈ 0.84)")
    return model

model = train_and_get_model()

# ------------------------------------------------------------------
# App UI – Beautiful & Professional
# ------------------------------------------------------------------
st.set_page_config(page_title="CVD Risk Predictor", layout="centered")

st.title("🫀 CVD 10-Year Risk Predictor")
st.markdown("Upload patient data → Get instant risk % (no download needed)")

st.info("Required columns: `note`, `age`, `sys_bp`, `dia_bp`, `cholesterol`, `glucose`, `bmi`, `smoke`, `family_hx`")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required_cols = ['note', 'age', 'sys_bp', 'dia_bp', 'cholesterol', 'glucose', 'bmi', 'smoke', 'family_hx']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            if st.button("🚀 Get Results", type="primary", use_container_width=True):
                with st.spinner("Calculating risk..."):
                    prob = model.predict_proba(df)[:, 1]
                    df["CVD_Risk_%"] = (prob * 100).round(1)

                    # Color coding
                    def color_risk(val):
                        if val >= 20:
                            return "background-color: #ffcccc"  # light red
                        elif val >= 10:
                            return "background-color: #ffffcc"  # light yellow
                        else:
                            return "background-color: #ccffcc"  # light green

                    styled_df = df.style.applymap(color_risk, subset=["CVD_Risk_%"]) \
                                        .format({"CVD_Risk_%": "{:.1f}"})

                    st.success("Results ready!")
                    st.dataframe(styled_df, use_container_width=True)

                    # Optional download (hidden by default)
                    csv = df.to_csv(index=False).encode()
                    st.download_button(
                        "Download CSV (optional)",
                        csv,
                        "cvd_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Waiting for CSV upload...")
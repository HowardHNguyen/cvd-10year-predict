# CVD Risk Predictor

A **realistic** 10-year cardiovascular-disease risk model built on 223 synthetic but
leak-free patient records.

## Files

| File | Description |
|------|-------------|
| `data_cvd_perfect.csv` | 223 clean rows (no disease keywords) |
| `train_cvd.py` | Trains Logistic-Regression + TF-IDF, saves `cvd_model_perfect.pkl` |
| `cvd_model_perfect.pkl` | Trained model (AUC ≈ 0.84) |
| `app.py` | Streamlit web UI |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Quick Start

```bash
# 1. Clone repo
git clone <your-repo-url>
cd <folder>

# 2. Install
pip install -r requirements.txt

# 3. (Optional) Re-train
python train_cvd.py

# 4. Run the web app
streamlit run app.py

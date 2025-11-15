# train_cvd.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv("data_cvd_perfect.csv")
print(f"Loaded {len(df)} PERFECT rows")

# ------------------------------------------------------------------
# 2. Train / test split
# ------------------------------------------------------------------
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df['cvd'], random_state=42
)

# ------------------------------------------------------------------
# 3. Feature columns
# ------------------------------------------------------------------
num_cols = ['age', 'sys_bp', 'dia_bp', 'cholesterol',
            'glucose', 'bmi', 'smoke', 'family_hx']
text_col = 'note'

# ------------------------------------------------------------------
# 4. Pre-processor (numeric + TF-IDF)
# ------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('text', TfidfVectorizer(max_features=20,
                             stop_words='english',
                             ngram_range=(1, 2)), text_col)
])

# ------------------------------------------------------------------
# 5. Model pipeline
# ------------------------------------------------------------------
model = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(C=0.4,
                               class_weight='balanced',
                               max_iter=1000))
])

# ------------------------------------------------------------------
# 6. Train
# ------------------------------------------------------------------
model.fit(train_df, train_df['cvd'])

# ------------------------------------------------------------------
# 7. Evaluate on hold-out set
# ------------------------------------------------------------------
pred = model.predict_proba(test_df)[:, 1]
auc = roc_auc_score(test_df['cvd'], pred)
acc = accuracy_score(test_df['cvd'], (pred > 0.5).astype(int))

print(f"\nFINAL REAL AUC: {auc:.4f} | Accuracy: {acc:.4f}")

# ------------------------------------------------------------------
# 8. Save model
# ------------------------------------------------------------------
joblib.dump(model, "cvd_model_perfect.pkl")
print("Saved → cvd_model_perfect.pkl")
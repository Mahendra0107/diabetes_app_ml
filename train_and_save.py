"""
Run this script ONCE locally to train and save all models.
Then push the saved .pkl files to GitHub along with app.py.

Usage:
    python train_and_save.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
SAVE_DIR     = "models"
DATA_FILE    = "diabetes.csv"   # put your diabetes.csv in the same folder

# ── Load Data ───────────────────────────────────────────────────────────────────
print(f"📂 Loading {DATA_FILE}...")
data = pd.read_csv(DATA_FILE)
print(f"   Shape: {data.shape}")

X = data.drop('Diabetes_012', axis=1)
y = data['Diabetes_012']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"   Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ── Train Models ─────────────────────────────────────────────────────────────────
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "decision_tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
    "random_forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
}

os.makedirs(SAVE_DIR, exist_ok=True)
results_summary = {}

for name, model in models.items():
    print(f"\n🧠 Training {name.replace('_', ' ').title()}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_test, preds)

    results_summary[name] = {
        "accuracy": acc,
        "report":   report,
        "cm":       cm.tolist(),
    }
    print(f"   ✅ Accuracy: {acc:.4f}")

    # Save model
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"   💾 Saved → {path}")

# ── Save metadata ────────────────────────────────────────────────────────────────
meta = {
    "feature_names": feature_names,
    "test_size":     TEST_SIZE,
    "random_state":  RANDOM_STATE,
    "train_rows":    int(X_train.shape[0]),
    "test_rows":     int(X_test.shape[0]),
    "total_rows":    int(len(data)),
    "results":       results_summary,
    "class_counts":  data['Diabetes_012'].value_counts().sort_index().to_dict(),
}
meta_path = os.path.join(SAVE_DIR, "metadata.pkl")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f)
print(f"\n📊 Metadata saved → {meta_path}")

print("\n✅ All done! Push the 'models/' folder to GitHub.")
print("   Files created:")
for fname in os.listdir(SAVE_DIR):
    fsize = os.path.getsize(os.path.join(SAVE_DIR, fname)) / 1024
    print(f"     {SAVE_DIR}/{fname}  ({fsize:.1f} KB)")

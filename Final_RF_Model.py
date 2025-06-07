# RF_Threshold_Module_2.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load prepared ML dataset
file = "SPY_ML_ready.csv"
df = pd.read_csv(file, index_col=0, parse_dates=True)
print(f"✅ Data loaded: {df.shape}")

# Features and labels
X = df.drop("Signal", axis=1)
y = df["Signal"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# Train RF model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Evaluate
y_pred = rf.predict(X_test_scaled)
print("🎯 RF Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Predict all
y_full_pred = rf.predict(X_scaled)
y_full_proba = rf.predict_proba(X_scaled)
y_conf = y_full_proba.max(axis=1)

# Save predictions + confidence
reverse_map = {-1: -1, 0: 0, 1: 1}
df["RF_Signal"] = y_full_pred
for i, cls in enumerate(rf.classes_):
    df[f"RF_Prob_{cls}"] = y_full_proba[:, i]
df["RF_Prob"] = y_conf

# Save to file
df.to_csv("SPY_with_equity_signals.csv")
print("✅ RF predictions saved to 'SPY_with_equity_signals.csv'")

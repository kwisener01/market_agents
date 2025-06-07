# train_rf_model.py

import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# === Step 1: Load Prepared Data ===
df = pd.read_csv("SPY_ML_ready.csv")
features = ["RSI", "MACD", "MACD_Signal", "MA_9", "MA_21", "MA_50"]
X = df[features]
y = df["Signal"]

# === Step 2: Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 3: Train Random Forest Model ===
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# === Step 4: Save Model, Scaler, and Metadata ===
with open("rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model_metadata.pkl", "wb") as f:
    pickle.dump({"features": features}, f)

print("\u2705 RF model, scaler, and metadata saved.")

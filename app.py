import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime
import pytz
import joblib
import gdown

#1ntpn9nsBHZ_jawfi7t_DiDGswIPhf_Dm
# --- Load Model + Config ---
# --- Load RF model from Google Drive using gdown ---
@st.cache_resource
def load_model_from_drive(file_id, output_path="rf_model.pkl"):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    return joblib.load(output_path)
# Replace with your actual file ID
model_file_id = "1ntpn9nsBHZ_jawfi7t_DiDGswIPhf_Dm"
rf_model = load_model_from_drive(model_file_id)


with open("rf_features.pkl", "rb") as f:
    FEATURES = pickle.load(f)
with open("rf_config.pkl", "rb") as f:
    CONFIG = pickle.load(f)

CONF_THRESH = CONFIG["confidence_threshold"]

# --- Live Data Fetching ---
def fetch_live_data(symbol, interval="1min", source="twelve"):
    if source == "twelve":
        key = st.secrets["api"]["twelve_data_key"]
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={key}&outputsize=1"
        r = requests.get(url).json()
        df = pd.DataFrame(r["values"])
        df = df.rename(columns=str.lower).astype(float)
        df["datetime"] = pd.to_datetime(r["values"][0]["datetime"])
        df.set_index("datetime", inplace=True)
        return df

# --- Feature Engineering (example placeholders) ---
def add_indicators(df):
    df["RSI"] = df["close"].rolling(14).mean()  # Replace with actual RSI
    df["MACD"] = df["close"].ewm(12).mean() - df["close"].ewm(26).mean()
    df["MACD_Signal"] = df["MACD"].ewm(9).mean()
    df["MA_9"] = df["close"].rolling(9).mean()
    df["MA_21"] = df["close"].rolling(21).mean()
    df["MA_50"] = df["close"].rolling(50).mean()
    return df.dropna()

# --- Prediction Logic ---
def predict(df):
    df = add_indicators(df)
    latest = df.iloc[[-1]]
    X_live = latest[FEATURES]
    prob = model.predict_proba(X_live)
    pred = np.argmax(prob)
    confidence = np.max(prob)
    signal_map = {0: -1, 1: 0, 2: 1}
    return signal_map[pred], confidence

# --- Streamlit App ---
st.title("📈 Real-Time SPY Buy/Sell/Hold Signals")
st.markdown("Powered by Random Forest & Twelve Data")

if st.button("📡 Fetch Live Signal"):
    try:
        live_df = fetch_live_data("SPY")
        signal, confidence = predict(live_df)
        label = {1: "🟢 BUY", 0: "⚪ HOLD", -1: "🔴 SELL"}[signal]

        st.metric("Signal", label)
        st.metric("Confidence", f"{confidence:.2%}")

        if confidence >= CONF_THRESH:
            st.success(f"✅ High confidence: {label}")
        else:
            st.warning(f"⚠️ Low confidence: {label}")
    except Exception as e:
        st.error(f"Error fetching or predicting live data: {e}")

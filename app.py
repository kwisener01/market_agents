# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import StandardScaler

# Load model, scaler, and metadata
@st.cache_resource
def load_model():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, scaler, metadata

rf_model, scaler, metadata = load_model()

# Get live SPY 1-minute data from Twelve Data
def fetch_spy_data():
    api_key = st.secrets["TWELVE_API_KEY"]
    url = f"https://api.twelvedata.com/time_series?symbol=SPY&interval=1min&outputsize=100&apikey={api_key}&format=JSON"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        st.error("Error fetching data from Twelve Data.")
        return None

    df = pd.DataFrame(data["values"])
    df = df.rename(columns={
        "datetime": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime")
    df.set_index("Datetime", inplace=True)
    df = df.astype(float)
    return df

# Feature engineering to match training set
def add_features(df):
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    macd = MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MA_9"] = df["Close"].rolling(window=9).mean()
    df["MA_21"] = df["Close"].rolling(window=21).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    return df.dropna()

# Prediction function
def predict(df):
    X = df[metadata["features"]]
    X_scaled = scaler.transform(X)
    proba = rf_model.predict_proba(X_scaled)
    signal = rf_model.predict(X_scaled)
    confidence = np.max(proba, axis=1)
    return signal[-1], confidence[-1]

# Streamlit UI
st.set_page_config(page_title="ClarityTrader Live Signal", layout="wide")
st.title("🧠 ClarityTrader - Live RF Signal")

with st.spinner("Fetching live SPY data..."):
    df = fetch_spy_data()

if df is not None:
    df = add_features(df)
    if not df.empty:
        signal, confidence = predict(df)
        signal_str = "Buy" if signal == 1 else "Sell" if signal == -1 else "Hold"

        st.metric("📉 Last Price", f"${df['Close'].iloc[-1]:.2f}")
        st.metric("🔍 Signal", signal_str)
        st.metric("🔒 Confidence", f"{confidence:.2%}")

        st.line_chart(df[["Close", "MA_9", "MA_21", "MA_50"]].tail(50))
    else:
        st.error("Not enough data for feature calculation.")

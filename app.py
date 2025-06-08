import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime
import pytz
import joblib
import gdown
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

# --- Load RF model from Google Drive using gdown ---
@st.cache_resource
def load_model_from_drive(file_id, output_path="rf_model.pkl"):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    return joblib.load(output_path)

# Load the model
model_file_id = "1ntpn9nsBHZ_jawfi7t_DiDGswIPhf_Dm"
rf_model = load_model_from_drive(model_file_id)

# Load metadata
with open("rf_features.pkl", "rb") as f:
    FEATURES = pickle.load(f)
with open("rf_config.pkl", "rb") as f:
    CONFIG = pickle.load(f)

CONF_THRESH = CONFIG["confidence_threshold"]
API_KEY = st.secrets["TWELVE_DATA"]["API_KEY"]

# --- Live Data Fetching ---
@st.cache_data(ttl=60)
def fetch_live_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY}"
    response = requests.get(url).json()
    if 'values' not in response:
        st.error(f"API Error: {response.get('message', 'Unknown')}")
        return None
    df = pd.DataFrame(response['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').set_index('datetime')
    df.columns = [c.lower() for c in df.columns]
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns from Twelve Data: {set(required_cols) - set(df.columns)}")
        return None
    df[required_cols] = df[required_cols].astype(float)
    return df

# --- Feature Engineering ---
def add_indicators(df):
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MA_9"] = df["close"].rolling(9).mean()
    df["MA_21"] = df["close"].rolling(21).mean()
    df["MA_50"] = df["close"].rolling(50).mean()
    return df.dropna()

# --- Prediction Logic ---
def predict(df):
    df = add_indicators(df)
    if df.shape[0] == 0:
        raise ValueError("Not enough data after indicator calculation. Need at least 50 rows.")
    latest = df.iloc[[-1]]
    X_live = latest[FEATURES]
    prob = rf_model.predict_proba(X_live)
    pred = np.argmax(prob)
    confidence = np.max(prob)
    signal_map = {0: -1, 1: 0, 2: 1}
    return signal_map[pred], confidence, df

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("📈 Real-Time SPY Buy/Sell/Hold Signals")
st.markdown("Powered by Random Forest & Twelve Data")

refresh_rate = st.selectbox("🔄 Auto-Refresh Rate:", ["Off", "30 sec", "60 sec"], index=2)
interval_map = {"Off": 0, "30 sec": 30, "60 sec": 60}
auto_refresh = interval_map[refresh_rate]
acknowledged = st.checkbox("✅ Acknowledge Signal Alert", value=False)

last_signal = st.empty()
last_confidence = st.empty()
alert_container = st.empty()

placeholder = st.empty()

while True:
    try:
        live_df = fetch_live_data("SPY", interval="1min")
        if live_df is not None:
            signal, confidence, live_df = predict(live_df)
            label = {1: "🟢 BUY", 0: "⚪ HOLD", -1: "🔴 SELL"}[signal]

            with placeholder.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    fig = go.Figure(data=[
                        go.Candlestick(
                            x=live_df.index,
                            open=live_df['open'],
                            high=live_df['high'],
                            low=live_df['low'],
                            close=live_df['close']
                        )
                    ])
                    fig.update_layout(title="Live SPY Candlestick Chart", xaxis_title="Time", yaxis_title="Price")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    last_signal.metric("Signal", label)
                    last_confidence.metric("Confidence", f"{confidence:.2%}")

                    if confidence >= CONF_THRESH:
                        st.success(f"✅ High confidence: {label}")
                        if not acknowledged:
                            alert_container.warning(f"🚨 Signal Alert: {label} (Click checkbox to acknowledge)")
                    else:
                        st.warning(f"⚠️ Low confidence: {label}")

    except Exception as e:
        st.error(f"Error fetching or predicting live data: {e}")

    if auto_refresh == 0:
        break
    time.sleep(auto_refresh)
    st.rerun()

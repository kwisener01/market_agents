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
import io

# --- Set page config FIRST ---
st.set_page_config(layout="wide")

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
AV_KEY = st.secrets["ALPHA_VANTAGE"]["API_KEY"]

# --- Session State for History ---
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()

# --- Refresh Interval Control ---
refresh_options = {
    "Every 1 minute": 60,
    "Every 2 minutes": 120,
    "Every 5 minutes": 300,
    "Manual": None
}
refresh_choice = st.selectbox("🔄 Auto-refresh interval:", list(refresh_options.keys()), index=0)
refresh_interval = refresh_options[refresh_choice]

# --- Live Data Fetching ---
@st.cache_data(ttl=60)
def fetch_live_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY}"
    response = requests.get(url).json()
    if 'values' not in response:
        st.error(f"API Error: {response.get('message', 'Unknown')}")
        return None
    df = pd.DataFrame(response['values'])
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df = df.sort_values('datetime').set_index('datetime')
    df.columns = [c.lower() for c in df.columns]
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns from Twelve Data: {set(required_cols) - set(df.columns)}")
        return None
    df[required_cols] = df[required_cols].astype(float)
    return df

@st.cache_data(ttl=300)
def fetch_alphavantage_data(symbol="SPY", interval="1min"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize=full&apikey={AV_KEY}&datatype=csv"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("AlphaVantage API Error")
        return None
    df = pd.read_csv(io.StringIO(response.text))
    if df.empty:
        st.warning("AlphaVantage returned no data.")
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df = df.rename(columns={"timestamp": "datetime"}).set_index("datetime")
    return df.sort_index()

# --- Feature Engineering ---
def add_indicators(df):
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=6).mean()
    avg_loss = pd.Series(loss).rolling(window=6).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["MACD"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
    df["MA_9"] = df["close"].rolling(6).mean()
    df["MA_21"] = df["close"].rolling(12).mean()
    df["MA_50"] = df["close"].rolling(20).mean()
    return df.dropna()

# --- Prediction Logic ---
def predict(df):
    raw_rows = df.shape[0]
    df = add_indicators(df)
    available_rows = df.shape[0]
    st.info(f"📊 Raw rows: {raw_rows}, Rows after indicators: {available_rows}, Required: 50")

    if available_rows == 0:
        raise ValueError("No valid rows after indicator calculation.")
    elif available_rows < 50:
        raise ValueError(f"Not enough data after indicator calculation. Got {available_rows}, need at least 50 rows.")

    latest = df.iloc[[-1]]
    X_live = latest[FEATURES]
    prob = rf_model.predict_proba(X_live)
    pred = np.argmax(prob)
    confidence = np.max(prob)
    signal_map = {0: -1, 1: 0, 2: 1}
    return signal_map[pred], confidence, df

# --- Quick AlphaVantage model test before market ---
if st.sidebar.button("🧪 Test Model with AlphaVantage"):
    try:
        test_df = fetch_alphavantage_data("SPY", interval="1min")
        if test_df is not None:
            signal, confidence, _ = predict(test_df)
            label = {1: "🟢 BUY", 0: "⚪ HOLD", -1: "🔴 SELL"}[signal]
            st.sidebar.success(f"Model OK — {label} with {confidence:.2%} confidence")
    except Exception as e:
        st.sidebar.error(f"❌ Model test failed: {e}")

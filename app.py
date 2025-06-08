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
    df = add_indicators(df)
    available_rows = df.shape[0]
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

# --- AlphaVantage Candlestick Chart ---
with st.expander("📉 AlphaVantage Candlestick Chart"):
    alpha_df = fetch_alphavantage_data("SPY", interval="1min")
    if alpha_df is not None and not alpha_df.empty:
        fig_alpha = go.Figure(data=[
            go.Candlestick(
                x=alpha_df.index,
                open=alpha_df["open"],
                high=alpha_df["high"],
                low=alpha_df["low"],
                close=alpha_df["close"],
                name="AlphaVantage"
            )
        ])
        fig_alpha.update_layout(title="SPY - AlphaVantage Candlestick Chart", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig_alpha, use_container_width=True)
    else:
        st.warning("No AlphaVantage data available.")

# --- Live Signal Section ---
st.header("📡 Real-Time Signal")
col1, col2 = st.columns([2, 1])

with col1:
    now = time.time()
    if refresh_interval is not None and now - st.session_state.last_refresh >= refresh_interval:
        st.session_state.last_refresh = now
    df_live = fetch_live_data("SPY", interval="1min")
    if df_live is not None:
        fig = go.Figure(data=[
            go.Candlestick(
                x=df_live.index,
                open=df_live["open"],
                high=df_live["high"],
                low=df_live["low"],
                close=df_live["close"],
                name="Live"
            )
        ])
        fig.update_layout(title="Live SPY Chart (Twelve Data)", xaxis_title="Time", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    if st.button("Run Model"):
        try:
            signal, confidence, full_df = predict(df_live)
            signal_label = {1: "🟢 BUY", 0: "⚪ HOLD", -1: "🔴 SELL"}[signal]
            st.metric("Signal", signal_label)
            st.metric("Confidence", f"{confidence:.2%}")

            # Save history
            now = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d %H:%M")
            st.session_state.signal_history.append({"time": now, "signal": signal_label, "confidence": confidence})

        except Exception as e:
            st.error(f"Error running model: {e}")

# --- Signal History ---
if st.session_state.signal_history:
    st.subheader("📜 Signal History")
    hist_df = pd.DataFrame(st.session_state.signal_history)
    st.dataframe(hist_df, use_container_width=True)

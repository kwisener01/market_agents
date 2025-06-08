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
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize=compact&apikey={AV_KEY}&datatype=csv"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("AlphaVantage API Error")
        return None
    df = pd.read_csv(io.StringIO(response.text))
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df = df.rename(columns={"timestamp": "datetime"}).set_index("datetime")
    return df.sort_index()

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

# --- Streamlit App ---
st.title("📈 Real-Time SPY Buy/Sell/Hold Signals")
st.markdown("Powered by Random Forest, Twelve Data & AlphaVantage")

refresh_rate = st.selectbox("🔄 Auto-Refresh Rate:", ["Off", "60 sec", "2 min", "5 min"], index=1)
interval_map = {"Off": 0, "60 sec": 60, "2 min": 120, "5 min": 300}
auto_refresh = interval_map[refresh_rate]
acknowledged = st.checkbox("✅ Acknowledge Signal Alert", value=False)

col1, col2 = st.columns([3, 1])

# --- Chart Area ---
with col1:
    st.subheader("📊 Live SPY Candlestick Chart (Twelve Data)")
    chart_placeholder = st.empty()

# --- Signal Area ---
with col2:
    st.subheader("🔍 Latest Prediction")
    if st.button("▶️ Run Model"):
        try:
            live_df = fetch_live_data("SPY", interval="1min")
            if live_df is not None:
                st.info(f"📊 Live rows available before indicators: {len(live_df)}")
                signal, confidence, live_df = predict(live_df)
                label = {1: "🟢 BUY", 0: "⚪ HOLD", -1: "🔴 SELL"}[signal]

                st.metric("Signal", label)
                st.metric("Confidence", f"{confidence:.2%}")

                if confidence >= CONF_THRESH:
                    st.success(f"✅ High confidence: {label}")
                    if not acknowledged:
                        st.warning(f"🚨 Signal Alert: {label} (Acknowledge below)")

                st.session_state.signal_history.append({
                    "time": datetime.now(pytz.timezone("America/New_York")).strftime("%H:%M:%S"),
                    "signal": label,
                    "confidence": confidence
                })

        except Exception as e:
            st.error(f"Error fetching or predicting live data: {e}")

    if st.session_state.signal_history:
        st.markdown("---")
        st.markdown("## 📜 Signal History")
        st.dataframe(pd.DataFrame(st.session_state.signal_history)[::-1])

# --- Auto-refresh Chart ---
while auto_refresh > 0:
    try:
        live_df = fetch_live_data("SPY", interval="1min")
        if live_df is not None:
            fig = go.Figure(data=[
                go.Candlestick(
                    x=live_df.index,
                    open=live_df['open'],
                    high=live_df['high'],
                    low=live_df['low'],
                    close=live_df['close']
                )
            ])
            fig.update_layout(title="Live SPY Chart (Twelve Data)", xaxis_title="Time", yaxis_title="Price")
            chart_placeholder.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Chart update error: {e}")
    time.sleep(auto_refresh)
    st.rerun()

# --- Tabs for AlphaVantage Chart + Stats ---
tab1, tab2 = st.tabs(["📉 AlphaVantage Chart", "📊 AlphaVantage Stats"])

with tab1:
    alpha_df = fetch_alphavantage_data("SPY", interval="1min")
    if alpha_df is not None:
        alpha_df = add_indicators(alpha_df)
        alpha_df = alpha_df.dropna()
        if not alpha_df.empty:
            latest_row = alpha_df.iloc[[-1]]
            pred, conf, _ = predict(alpha_df)
            signal_map = {-1: "SELL", 0: "HOLD", 1: "BUY"}
            alpha_df["Signal"] = ""
            alpha_df.loc[latest_row.index, "Signal"] = signal_map[pred]

            fig_hist = go.Figure(data=[
                go.Candlestick(
                    x=alpha_df.index,
                    open=alpha_df['open'],
                    high=alpha_df['high'],
                    low=alpha_df['low'],
                    close=alpha_df['close']
                )
            ])
            fig_hist.update_layout(title="AlphaVantage SPY 1-Min Historical Chart", xaxis_title="Time", yaxis_title="Price")
            st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    if alpha_df is not None:
        st.write("## Descriptive Statistics")
        st.dataframe(alpha_df.describe())

        if "Signal" in alpha_df.columns:
            signal_counts = alpha_df["Signal"].value_counts()
            st.write("## Signal Distribution")
            st.bar_chart(signal_counts)

            # Model ROI Simulation
            prices = alpha_df["close"].values
            signals = alpha_df["Signal"].replace({"BUY": 1, "SELL": -1, "HOLD": 0}).values
            initial_cash = 100000
            pos, cash = 0, initial_cash
            for i in range(len(signals)):
                if signals[i] == 1 and pos == 0:
                    pos = cash / prices[i]
                    cash = 0
                elif signals[i] == -1 and pos > 0:
                    cash = pos * prices[i]
                    pos = 0
            final_equity = cash + pos * prices[-1]
            roi = (final_equity - initial_cash) / initial_cash * 100
            st.metric("Backtest ROI", f"{roi:.2f}%")

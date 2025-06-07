import streamlit as st
import pandas as pd
import requests
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
import mplfinance as mpf
import matplotlib.pyplot as plt
import openai
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import yfinance as yf
import pytz
from datetime import datetime

API_KEY = st.secrets["TWELVE_DATA"]["API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI"]["API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

SYMBOL = st.selectbox("Select Symbol", ["SPY", "QQQ", "AAPL", "TSLA"], index=0)
INTERVAL = "1min"

refresh_rate = st.selectbox("\u23f1\ufe0f Auto-Refresh Interval", ["Do not refresh", "1 min", "2 min", "5 min"], index=1)
interval_mapping = {"Do not refresh": 0, "1 min": 60 * 1000, "2 min": 120 * 1000, "5 min": 300 * 1000}
interval_ms = interval_mapping[refresh_rate]
if interval_ms > 0:
    st_autorefresh(interval=interval_ms, key="autorefresh")

shared_memory = {}

@st.cache_data(ttl=60)
def fetch_data(symbol, interval):
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

def generate_signals(df):
    df.columns = [c.lower() for c in df.columns]
    if 'close' not in df.columns:
        raise ValueError("Missing 'close' column after column standardization.")
    df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    df['ema_20'] = df['close'].ewm(span=20).mean()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    bb = BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    vwap = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'])
    df['vwap'] = vwap.volume_weighted_average_price()
    df['signal'] = 0
    df.loc[(df['rsi'] < 30) & (df['close'] > df['ema_20']), 'signal'] = 1
    df.loc[(df['rsi'] > 70) & (df['close'] < df['ema_20']), 'signal'] = -1
    df.loc[(df['rsi'].between(40, 60)) & (df['signal'] == 0), 'signal'] = 0
    return df

def plot_chart(df):
    plot_df = df[-100:].copy()
    apds = [
        mpf.make_addplot(plot_df['ema_20'], color='blue'),
        mpf.make_addplot(plot_df['rsi'], panel=1, ylabel='RSI'),
        mpf.make_addplot(plot_df['signal'], type='scatter', markersize=100, marker='^', color='green', panel=0),
        mpf.make_addplot(-plot_df['signal'], type='scatter', markersize=100, marker='v', color='red', panel=0)
    ]
    fig, _ = mpf.plot(plot_df, type='candle', style='charles', addplot=apds, volume=True, returnfig=True)
    return fig

def train_predictive_model(df):
    df = generate_signals(df)
    df = df.dropna().copy()
    df.columns = [c.lower() for c in df.columns]
    df['label'] = df['signal'].map({1: "Buy", -1: "Sell", 0: "Hold"})
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=10).std()
    df['volume_surge'] = df['volume'] / df['volume'].rolling(10).mean()
    features = ["rsi", "ema_20", "macd", "macd_signal", "bb_upper", "bb_lower", "vwap", "price_change", "volatility", "volume_surge"]
    df = df.dropna(subset=features + ['label'])

    st.write("🔍 Data preview before training:", df.head())

    if df.empty or len(df) < 50:
        st.warning(f"⚠️ Not enough valid training data after feature processing. Found: {len(df)} rows.")
        return None

    X = df[features]
    y = df['label']
    st.write("📐 Feature matrix shape:", X.shape, "| Target shape:", y.shape)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    shared_memory['features'] = features
    shared_memory['model'] = model
    shared_memory['last_df'] = df
    df.to_csv("prediction_history.csv")

    if os.path.exists("model.pkl"):
        st.success("✅ model.pkl successfully written to disk.")
    else:
        st.error("❌ model.pkl was not created. Training may have failed.")
    return model

# ...rest of code remains unchanged...

if st.button("Train Model on Yahoo Finance"):
    try:
        hist = yf.download(SYMBOL, period="7d", interval="1m")
        if hist is None or hist.empty:
            raise ValueError("Yahoo Finance returned no data.")

        hist.columns = [col.lower().strip().replace(" ", "_") for col in hist.columns]
        hist = hist.dropna()
        hist.index.name = "datetime"
        hist = hist.reset_index().set_index("datetime")

        st.info(f"✅ Yahoo data shape after cleanup: {hist.shape}")

        if 'close' not in hist.columns:
            raise ValueError(f"No 'close' column found in Yahoo data after cleanup. Columns available: {list(hist.columns)}")

        hist = generate_signals(hist)
        trained_model = train_predictive_model(hist)
        if trained_model:
            st.success("✅ Model trained and saved as model.pkl.")

    except Exception as e:
        st.error(f"Yahoo Training Error: {e}")

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
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def generate_signals(df):
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
    df['Label'] = df['signal'].map({1: "Buy", -1: "Sell", 0: "Hold"})
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=10).std()
    df['volume_surge'] = df['volume'] / df['volume'].rolling(10).mean()
    features = ["rsi", "ema_20", "macd", "macd_signal", "bb_upper", "bb_lower", "vwap", "price_change", "volatility", "volume_surge"]
    df = df.dropna(subset=features + ['Label'])
    if df.empty or len(df) < 50:
        st.warning(f"\u26a0\ufe0f Not enough valid training data after feature processing. Found: {len(df)} rows.")
        return None
    X = df[features]
    y = df['Label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    shared_memory['features'] = features
    shared_memory['model'] = model
    shared_memory['last_df'] = df
    df.to_csv("prediction_history.csv")
    return model

def display_prediction(df, model):
    if model is None:
        st.error("⚠️ No model available. Please train one.")
        return
    features = shared_memory.get('features')
    if not features or not set(features).issubset(df.columns):
        st.warning("⚠️ Required features are missing in data.")
        return
    latest = df.iloc[-1:]
    X_latest = latest[features]
    pred = model.predict(X_latest)[0]
    price = latest['close'].values[0]
    now = datetime.now(pytz.timezone("US/Eastern"))
    st.metric("Current Price", f"${price:.2f}")
    st.metric("Current Time (EST)", now.strftime("%I:%M %p"))
    st.metric("Suggested Action", pred)

def bayesian_agent(df):
    row = df.iloc[-1]
    prompt = f"Given RSI={row['rsi']:.2f}, MACD={row['macd']:.2f}, and Price={row['close']:.2f}, use Bayesian thinking to say Buy, Hold, or Sell."
    res = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    st.subheader("🧠 Bayesian Agent")
    st.write(res.choices[0].message.content)

def ml_insight_agent():
    suggestion = "Try feature selection with SHAP or permutation importance. Consider seasonality, open-close gaps, or time of day as features."
    shared_memory['ml_insight'] = suggestion
    st.subheader("📊 ML Insight Agent")
    st.write(suggestion)

def market_summary_agent():
    summary = f"Market looks stable for {SYMBOL}. No major news alerts."
    shared_memory['market_summary'] = summary
    st.subheader("🌐 Market Agent")
    st.write(summary)

def manager_agent():
    summary = f"Model trained. Using features: {shared_memory.get('features', [])}\nML Suggestions: {shared_memory.get('ml_insight', '')}\nMarket Summary: {shared_memory.get('market_summary', '')}"
    st.subheader("🧠 Manager Agent")
    st.code(summary)

if os.path.exists("model.pkl"):
    model = joblib.load("model.pkl")
    shared_memory['model'] = model
else:
    model = None

data = fetch_data(SYMBOL, INTERVAL)
if data is not None:
    signals = generate_signals(data)
    fig = plot_chart(signals)
    st.pyplot(fig)
    display_prediction(signals, shared_memory.get('model'))

    if st.button("Run Bayesian Agent"):
        bayesian_agent(signals)
    if st.button("Run ML Insight Agent"):
        ml_insight_agent()
    if st.button("Run Market Summary Agent"):
        market_summary_agent()
    if st.button("Run Manager Agent"):
        manager_agent()

if st.button("Train Model on Yahoo Finance"):
    try:
        hist_result = yf.download(SYMBOL, period="7d", interval="1m")
        hist = hist_result[0] if isinstance(hist_result, tuple) else hist_result
        hist = hist.dropna()
        hist.columns = [str(c).replace(" ", "_").lower() for c in hist.columns]
        hist.index.name = "datetime"
        hist = hist.rename(columns={"adj_close": "close"})
        hist = hist.reset_index().set_index("datetime")
        hist = generate_signals(hist)
        trained_model = train_predictive_model(hist)
        if trained_model:
            st.success("✅ Model trained and saved as model.pkl.")
    except Exception as e:
        st.error(f"Yahoo Training Error: {e}")

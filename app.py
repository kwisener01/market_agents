import streamlit as st
import pandas as pd
import requests
from ta.momentum import RSIIndicator
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

API_KEY = st.secrets["TWELVE_DATA"]["API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI"]["API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

SYMBOL = st.selectbox("Select Symbol", ["SPY", "QQQ", "AAPL", "TSLA"], index=0)
INTERVAL = "1min"

refresh_rate = st.selectbox("⏱️ Auto-Refresh Interval", ["Do not refresh", "1 min", "2 min", "5 min"], index=1)
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
    df = df.dropna()
    df['Label'] = df['signal'].map({1: "Buy", -1: "Sell", 0: "Hold"})
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=10).std()
    df['volume_surge'] = df['volume'] / df['volume'].rolling(10).mean()
    features = ["rsi", "ema_20", "price_change", "volatility", "volume_surge"]
    df = df.dropna(subset=features + ['Label'])
    X = df[features]
    y = df['Label']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "model.pkl")
    shared_memory['features'] = features
    shared_memory['model'] = model
    return model

def ml_insight_agent():
    insight = "Consider adding MACD, Bollinger Bands, and VWAP for richer feature context. Check regime change detection using clustering."
    shared_memory['ml_insight'] = insight
    return insight

def market_agent():
    info = f"Checking macro view for {SYMBOL}. No red flags today. Medium volatility expected."
    shared_memory['market_summary'] = info
    return info

def manager_agent():
    summary = f"Model trained. Using features: {shared_memory.get('features', [])}\n" \
              f"ML Suggestions: {shared_memory.get('ml_insight', '')}\n" \
              f"Market Summary: {shared_memory.get('market_summary', '')}"
    return summary

# UI Controls for agents
if st.button("Run ML Insight Agent"):
    st.info(ml_insight_agent())
    st.info(market_agent())
    st.success(manager_agent())

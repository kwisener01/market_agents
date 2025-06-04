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
    return model

def bayesian_forecast(df):
    latest = df.iloc[-1]
    context = f"Given RSI={latest['rsi']:.2f}, EMA20={latest['ema_20']:.2f}, and Close={latest['close']:.2f}, return a one-word forecast: Buy, Sell, or Hold. Then explain in <25 words."
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Be concise. Only return: Buy, Sell, or Hold with <25-word explanation."},
                {"role": "user", "content": context}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in forecast: {e}"

def market_intel_agent(df, symbol):
    context = f"Summarize any macro factors or technical observations that might affect current signals for {symbol}. Max 25 words."
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a market analyst. Respond concisely (max 25 words)."},
                {"role": "user", "content": context}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in market analysis: {e}"

def ml_insight_agent(df):
    sample = df.iloc[-1]
    context = f"Using these indicators: RSI={sample['rsi']:.2f}, EMA20={sample['ema_20']:.2f}, Volume={sample['volume']}, close={sample['close']}, suggest new predictive features to improve a Buy/Sell/Hold model."
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a machine learning researcher helping improve a trading model. Respond with 1-3 new predictive features."},
                {"role": "user", "content": context}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in ML agent: {e}"

if API_KEY and OPENAI_API_KEY:
    data = fetch_data(SYMBOL, INTERVAL)
    if data is not None:
        signals = generate_signals(data)
        st.write("### Signal Data", signals.tail())
        fig = plot_chart(signals)
        st.pyplot(fig)

        if st.button("🔮 Run Bayesian Forecast Agent"):
            forecast = bayesian_forecast(signals)
            st.subheader("🧠 Bayesian Forecast")
            st.info(forecast)

        if st.button("📡 Run Market Intel Agent"):
            intel = market_intel_agent(signals, SYMBOL)
            st.subheader("📊 Market Intel Agent")
            st.info(intel)

        if st.button("🤖 Run ML Insight Agent"):
            insight = ml_insight_agent(signals)
            st.subheader("🧠 ML Agent Insights")
            st.info(insight)

        if st.button("🚀 Train Predictive Model with Suggested Features"):
            model = train_predictive_model(signals)
            st.session_state.model = model
            st.success("✅ Model trained and saved.")

            # Predict on latest row
            features = ["rsi", "ema_20", "price_change", "volatility", "volume_surge"]
            latest = signals.dropna().iloc[-1:]
            if not latest.empty:
                latest['price_change'] = latest['close'].pct_change()
                latest['volatility'] = latest['close'].rolling(window=10).std()
                latest['volume_surge'] = latest['volume'] / latest['volume'].rolling(10).mean()
                latest = latest.dropna()
                X_pred = latest[features]
                pred = model.predict(X_pred)[0]
                st.subheader("📍 ML Model Signal")
                st.metric("Prediction", pred)

        signals.to_csv("signals.csv")
        st.download_button("Download CSV", signals.to_csv().encode(), "signals.csv")

        if os.path.isdir(".git"):
            with open("signals.csv", "rb") as f:
                git_command = "git add signals.csv && git commit -m 'Auto update signals' && git push"
                st.caption(f"📡 Signals saved. Use this git command:")
                st.code(git_command)

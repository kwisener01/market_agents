import streamlit as st
import pandas as pd
import requests
from ta.momentum import RSIIndicator
import mplfinance as mpf
import matplotlib.pyplot as plt
import openai
from openai import OpenAI

API_KEY = st.secrets["TWELVE_DATA"]["API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI"]["API_KEY"]

client = OpenAI(api_key=OPENAI_API_KEY)

SYMBOL = st.selectbox("Select Symbol", ["SPY", "QQQ", "AAPL", "TSLA"], index=0)
INTERVAL = "1min"

refresh_rate = st.selectbox("⏱️ Auto-Refresh Interval", ["Do not refresh", "1 min", "2 min", "5 min"], index=1)
interval_mapping = {"Do not refresh": 0, "1 min": 60 * 1000, "2 min": 120 * 1000, "5 min": 300 * 1000}
interval_ms = interval_mapping[refresh_rate]

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

if API_KEY and OPENAI_API_KEY:
    data = fetch_data(SYMBOL, INTERVAL)
    if data is not None:
        signals = generate_signals(data)
        st.write("### Signal Data", signals.tail())
        fig = plot_chart(signals)
        st.pyplot(fig)
        forecast = bayesian_forecast(signals)
        st.write("### 🧠 Bayesian Forecast")
        st.info(forecast)
        signals.to_csv("signals.csv")
        st.download_button("Download CSV", signals.to_csv().encode(), "signals.csv")

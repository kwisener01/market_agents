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

# --- Column name standardization function ---
def standardize_columns(df):
    """Standardize column names to match model expectations"""
    # Create a mapping for common column name variations
    column_mapping = {
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low', 
        'Close': 'Close',
        'Volume': 'Volume'
    }
    
    # Rename columns based on mapping
    df_renamed = df.rename(columns=column_mapping)
    
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df_renamed.columns]
    
    if missing_cols:
        st.error(f"Missing required columns after standardization: {missing_cols}")
        st.write("Available columns:", list(df_renamed.columns))
        return None
        
    return df_renamed

# --- Live Data Fetching ---
@st.cache_data(ttl=60)
def fetch_live_data(symbol, interval):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize=full&apikey={API_KEY}"
    response = requests.get(url).json()
    if 'values' not in response:
        st.error(f"API Error: {response.get('message', 'Unknown')}")
        return None
    df = pd.DataFrame(response['values'])
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df = df.sort_values('datetime').set_index('datetime')
    
    # Standardize column names
    df = standardize_columns(df)
    if df is None:
        return None
        
    # Convert to float
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    st.info(f"Fetched {len(df)} rows of data for {symbol}")
    return df

@st.cache_data(ttl=300)
def fetch_alphavantage_data(symbol="SPY", interval="1min"):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&outputsize=full&apikey={AV_KEY}&datatype=csv"
    response = requests.get(url)
    if response.status_code != 200:
        st.error("AlphaVantage API Error")
        return None
    if not response.text or not response.text.startswith("timestamp"):
        st.warning("Missing 'timestamp' column in AlphaVantage data.")
        st.text("Raw AlphaVantage response (preview):")
        st.code(response.text[:500])
        return None
    df = pd.read_csv(io.StringIO(response.text))
    if df.empty:
        st.warning("AlphaVantage returned no data.")
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    df = df.rename(columns={"timestamp": "datetime"}).set_index("datetime")
    df = df.sort_index()
    
    # Standardize column names for AlphaVantage
    df = standardize_columns(df)
    
    st.info(f"AlphaVantage fetched {len(df)} rows")
    return df

# --- Feature Engineering ---
def add_indicators(df):
    # Use Close column for calculations
    delta = df["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Adjust window sizes based on available data
    data_len = len(df)
    rsi_window = min(6, max(2, data_len // 5))  # Adaptive RSI window
    ma_short = min(6, max(2, data_len // 5))    # Adaptive short MA
    ma_med = min(12, max(3, data_len // 3))     # Adaptive medium MA
    ma_long = min(20, max(4, data_len // 2))    # Adaptive long MA
    
    st.info(f"Using adaptive windows: RSI={rsi_window}, MA_short={ma_short}, MA_med={ma_med}, MA_long={ma_long}")
    
    # RSI calculation with adaptive window
    avg_gain = pd.Series(gain).rolling(window=rsi_window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=rsi_window, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD with adaptive spans
    ema_fast = min(12, max(3, data_len // 3))
    ema_slow = min(26, max(5, data_len // 2))
    signal_span = min(9, max(2, data_len // 4))
    
    df["MACD"] = df["Close"].ewm(span=ema_fast, min_periods=1).mean() - df["Close"].ewm(span=ema_slow, min_periods=1).mean()
    df["MACD_Signal"] = df["MACD"].ewm(span=signal_span, min_periods=1).mean()
    
    # Moving averages with adaptive windows and min_periods
    df["MA_9"] = df["Close"].rolling(ma_short, min_periods=1).mean()
    df["MA_21"] = df["Close"].rolling(ma_med, min_periods=1).mean()
    df["MA_50"] = df["Close"].rolling(ma_long, min_periods=1).mean()

    # Check which features are actually needed by the model
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        st.warning(f"Missing columns required by model: {missing_cols}")
        st.write("Available columns:", list(df.columns))
        st.write("Required features:", FEATURES)
        raise ValueError(f"Missing columns required by model: {missing_cols}")

    # Check for NaN values in required features
    nan_counts = df[FEATURES].isna().sum()
    if nan_counts.sum() > 0:
        st.warning("NaN values found in features:")
        st.write(nan_counts[nan_counts > 0])
        
        # Fill remaining NaN values with forward fill then backward fill
        df[FEATURES] = df[FEATURES].fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN, fill with column means
        for col in FEATURES:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())

    return df.dropna(subset=FEATURES)

# --- Prediction Logic ---
def predict(df):
    raw_rows = df.shape[0]
    df = add_indicators(df)
    available_rows = df.shape[0]
    st.info(f"📊 Raw rows: {raw_rows}, Rows after indicators: {available_rows}")

    if available_rows == 0:
        raise ValueError("No valid rows after indicator calculation.")
    elif available_rows < 10:  # Reduced minimum requirement
        st.warning(f"Limited data available. Got {available_rows} rows. Predictions may be less reliable.")
    
    # Use the most recent available data
    latest = df.iloc[[-1]]
    X_live = latest[FEATURES]
    
    # Debug: Show the feature values
    st.write("Feature values for latest prediction:")
    feature_dict = {feature: X_live[feature].iloc[0] for feature in FEATURES}
    st.write(feature_dict)
    
    prob = rf_model.predict_proba(X_live)
    pred = np.argmax(prob)
    confidence = np.max(prob)
    signal_map = {0: -1, 1: 0, 2: 1}
    return signal_map[pred], confidence, df

# --- Debug Info ---
st.sidebar.header("🔧 Debug Info")
st.sidebar.write("Required Features:")
st.sidebar.write(FEATURES)

# --- Quick AlphaVantage model test before market ---
if st.sidebar.button("🧪 Test Model with AlphaVantage"):
    try:
        test_df = fetch_alphavantage_data("SPY", interval="1min")
        if test_df is not None:
            st.sidebar.write("AlphaVantage columns:", list(test_df.columns))
            signal, confidence, _ = predict(test_df)
            label = {1: "🟢 BUY", 0: "⚪ HOLD", -1: "🔴 SELL"}[signal]
            st.sidebar.success(f"Model OK — {label} with {confidence:.2%} confidence")
    except Exception as e:
        st.sidebar.error(f"❌ Model test failed: {e}")

# --- Real-time Display ---
st.header("📡 Real-Time Signal")
st.subheader("Live SPY Chart (Twelve Data)")
live_data = fetch_live_data("SPY", interval="1min")
if live_data is not None:
    st.write("Live data columns:", list(live_data.columns))
    fig = go.Figure(data=[
        go.Candlestick(
            x=live_data.index,
            open=live_data['Open'],
            high=live_data['High'],
            low=live_data['Low'],
            close=live_data['Close']
        )
    ])
    st.plotly_chart(fig, use_container_width=True)
    latest_price = live_data['Close'].iloc[-1]
    st.metric("Current Price", f"${latest_price:.2f}")

    # Bayesian forecast (simple example)
    st.subheader("🔮 Bayesian Forecast (Mean + 95% CI)")
    prices = live_data['Close'].values
    mean_price = np.mean(prices)
    std_dev = np.std(prices)
    ci_upper = mean_price + 1.96 * std_dev
    ci_lower = mean_price - 1.96 * std_dev
    st.write(f"Mean: ${mean_price:.2f} | 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

# --- Run Model ---
if st.button("▶️ Run Model"):
    try:
        if live_data is not None:
            signal, confidence, _ = predict(live_data)
            label = {1: "🟢 BUY", 0: "⚪ HOLD", -1: "🔴 SELL"}[signal]
            st.metric("Signal", label)
            st.metric("Confidence", f"{confidence:.2%}")
    except Exception as e:
        st.error(f"Error running model: {e}")

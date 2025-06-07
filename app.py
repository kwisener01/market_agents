# Final_RF_Model.py
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
from io import BytesIO
import streamlit as st

# --- Streamlit Integration ---
st.set_page_config(page_title="RF Model Equity Viewer", layout="wide")
st.title("📈 Random Forest Equity Curve Viewer")

# --- Load RF model from Google Drive ---
@st.cache_resource
def load_model_from_drive(file_id):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    return joblib.load(BytesIO(response.content))

# Replace with your actual file ID
model_file_id = "1_ZZSAU2P6H5_kVbp8RLudY8FRr0fSuQ6"
rf_model = load_model_from_drive(model_file_id)

# Load dataset with RF predictions and probabilities
df = pd.read_csv("SPY_with_equity_signals.csv", index_col=0, parse_dates=True)

# Apply filtered signal using best known config (confidence >= 0.75)
df["RF_Filtered"] = df.apply(lambda row: row["RF_Signal"] if row["RF_Prob"] >= 0.75 else 0, axis=1)

# Equity curve simulation (hold time & take-profit based)
def simulate_equity(signals, prices, hold=10, tp=0.005):
    equity = [100000]  # Starting capital
    position = 0
    entry_price = 0
    entry_time = None
    trades = 0

    index_list = prices.index.to_list()

    for i in range(1, len(signals)):
        current_time = index_list[i]
        price = prices.iloc[i]
        change = price - entry_price if entry_price != 0 else 0

        # Exit logic
        if position != 0 and entry_time is not None:
            elapsed_minutes = (current_time - entry_time).seconds // 60
            if (position == 1 and change >= tp) or (position == -1 and change <= -tp) or elapsed_minutes >= hold:
                position = 0
                entry_price = 0
                entry_time = None

        # Entry logic
        if position == 0:
            if signals.iloc[i] == 1:
                position = 1
                entry_price = price
                entry_time = current_time
                trades += 1
            elif signals.iloc[i] == -1:
                position = -1
                entry_price = price
                entry_time = current_time
                trades += 1

        prev_price = prices.iloc[i - 1]
        equity.append(equity[-1] * (1 + position * (price - prev_price) / prev_price))

    equity_series = pd.Series(equity[1:], index=signals.index[1:])
    return equity_series, trades

# Run simulation for best RF model
rf_curve, rf_trades = simulate_equity(df["RF_Filtered"], df["Close"], hold=10, tp=0.005)

# Plot equity curve
st.subheader("Equity Curve")
st.line_chart(rf_curve)

# Save results
df.loc[rf_curve.index, "EC_RF"] = rf_curve
df.to_csv("SPY_with_equity_signals.csv")

st.success("✅ Best RF equity curve computed and saved to 'SPY_with_equity_signals.csv'")
st.markdown(f"**📊 RF Trades:** {rf_trades}")

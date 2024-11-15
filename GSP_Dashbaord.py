import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(layout="wide", page_title="Stock DMI MACD States Dashboard")

# Portfolio definitions
default_stocks = {
    "HK Stocks": ["^HSI"] + [
        "0001.HK", "0003.HK", "0005.HK", "0006.HK", "0011.HK", "0012.HK", "0016.HK", "0017.HK",
        "0019.HK", "0020.HK", "0027.HK", "0066.HK", "0175.HK", "0241.HK", "0267.HK", "0268.HK",
        "0285.HK", "0288.HK", "0291.HK", "0293.HK", "0358.HK", "0386.HK", "0388.HK", "0522.HK",
        "0669.HK", "0688.HK", "0700.HK", "0762.HK", "0772.HK", "0799.HK", "0823.HK", "0836.HK",
        "0853.HK", "0857.HK", "0868.HK", "0883.HK", "0909.HK", "0914.HK", "0916.HK", "0939.HK",
        "0941.HK", "0960.HK", "0968.HK", "0981.HK", "0992.HK", "1024.HK", "1038.HK", "1044.HK",
        "1093.HK", "1109.HK", "1113.HK", "1177.HK", "1211.HK", "1299.HK", "1347.HK", "1398.HK",
        "1772.HK", "1776.HK", "1787.HK", "1801.HK", "1810.HK", "1818.HK", "1833.HK", "1876.HK",
        "1898.HK", "1928.HK", "1929.HK", "1997.HK", "2007.HK", "2013.HK", "2015.HK", "2018.HK",
        "2269.HK", "2313.HK", "2318.HK", "2319.HK", "2331.HK", "2333.HK", "2382.HK", "2388.HK",
        "2518.HK", "2628.HK", "3690.HK", "3888.HK", "3888.HK", "3968.HK", "6060.HK", "6078.HK",
        "6098.HK", "6618.HK", "6690.HK", "6862.HK", "9618.HK", "9626.HK", "9698.HK", "9888.HK",
        "9961.HK", "9988.HK", "9999.HK"
    ],
    "US Stocks": ["^NDX", "^SPX"] + [
        "AAPL", "ABBV", "ABNB", "ACN", "ADBE", "AMD", "AMGN", "AMZN", "AMT", "ASML",
        "AVGO", "BA", "BKNG", "BLK", "CAT", "CCL", "CDNS", "CEG", "CHTR", "COST", 
        "CRM", "CRWD", "CVS", "CVX", "DDOG", "DE", "DIS", "EQIX", "FTNT", "GE",
        "GILD", "GOOG", "GS", "HD", "IBM", "ICE", "IDXX", "INTC", "INTU", "ISRG",
        "JNJ", "JPM", "KO", "LEN", "LLY", "LRCX", "MA", "META", "MMM", "MRK", 
        "MS", "MSFT", "MU", "NEE", "NFLX", "NRG", "NVO", "NVDA", "OXY", "PANW",
        "PFE", "PG", "PGR", "PLTR", "PYPL", "QCOM", "REGN", "SBUX", "SMH", "SNOW",
        "SPGI", "TEAM", "TJX", "TRAV", "TSM", "TSLA", "TTD", "TXN", "UNH", "UPS",
        "V", "VST", "VZ", "WMT", "XOM", "ZS",
        "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLRE", "XLY"
    ],
    "World Index": [
        "^SPX", "^NDX", "^RUT", "^SOX", "^TNX", "^DJI", "^HSI", "3032.HK", "XIN9.FGI", 
        "^N225", "^BSESN", "^KS11", "^TWII", "^GDAXI", "^FTSE", "^FCHI", "^BVSP", "EEMA", 
        "EEM", "^HUI", "CL=F", "GC=F", "HG=F", "SI=F", "DX-Y.NYB", "BTC=F", "ETH=F"
    ]
}

TIMEFRAMES = {
    "Weekly": "1wk",
    "Daily": "1d",
    "Hourly": "1h"
}

# Technical Analysis Functions
def calculate_dmi(df, length=14, smoothing=14):
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate True Range
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Calculate directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate smoothed values
    tr_smoothed = pd.Series(tr).rolling(window=length).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=length).mean() / tr_smoothed
    minus_di = 100 * pd.Series(minus_dm).rolling(window=length).mean() / tr_smoothed
    
    # Calculate ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=smoothing).mean()
    
    return plus_di, minus_di, adx

def custom_ema(series, length, alpha_adj=19):
    alpha = 2 / (length + alpha_adj)
    return series.ewm(alpha=alpha, adjust=False).mean()

def calculate_macd(df, fast_length=12, slow_length=26, signal_length=9, alpha_adj=19):
    close = df['Close']
    fast_ma = custom_ema(close, fast_length, alpha_adj)
    slow_ma = custom_ema(close, slow_length, alpha_adj)
    macd = fast_ma - slow_ma
    signal = custom_ema(macd, signal_length, alpha_adj)
    return macd, signal

def get_state(plus_di, minus_di, adx):
    value = 0
    state = ""
    
    if plus_di.iloc[-1] > minus_di.iloc[-1]:
        value = 4 if adx.iloc[-1] > adx.iloc[-2] else 3
        state = "Bullish+" if adx.iloc[-1] > adx.iloc[-2] else "Bullish-"
    else:
        value = -4 if adx.iloc[-1] > adx.iloc[-2] else -3
        state = "Bearish+" if adx.iloc[-1] > adx.iloc[-2] else "Bearish-"
    
    return value, state

def set_state(macd):
    value = 0
    state = ""
    
    if macd.iloc[-1] > 0:
        value = 2 if macd.iloc[-1] > macd.iloc[-2] else 1
        state = "Bullish+" if macd.iloc[-1] > macd.iloc[-2] else "Bullish-"
    else:
        value = -2 if macd.iloc[-1] < macd.iloc[-2] else -1
        state = "Bearish+" if macd.iloc[-1] < macd.iloc[-2] else "Bearish-"
    
    return value, state

def go_state(signal):
    value = 0
    state = ""
    
    if signal.iloc[-1] > 0:
        value = 2 if signal.iloc[-1] > signal.iloc[-2] else 1
        state = "Bullish+" if signal.iloc[-1] > signal.iloc[-2] else "Bullish-"
    else:
        value = -2 if signal.iloc[-1] < signal.iloc[-2] else -1
        state = "Bearish+" if signal.iloc[-1] < signal.iloc[-2] else "Bearish-"
    
    return value, state

def get_trend(total_score):
    if total_score >= 5:
        return "Buy", "green"
    elif total_score <= -5:
        return "Sell", "red"
    else:
        return "Neutral", "white"

def fetch_data(symbol, timeframe):
    end_date = datetime.now()
    if timeframe == "1h":
        start_date = end_date - timedelta(days=7)
    elif timeframe == "1d":
        start_date = end_date - timedelta(days=100)
    else:  # Weekly
        start_date = end_date - timedelta(days=365)
    
    try:
        data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def analyze_symbol(data):
    if data is None or len(data) < 30:
        return None
    
    plus_di, minus_di, adx = calculate_dmi(data)
    macd, signal = calculate_macd(data)
    
    get_val, get_str = get_state(plus_di, minus_di, adx)
    set_val, set_str = set_state(macd)
    go_val, go_str = go_state(signal)
    
    total_score = get_val + set_val + go_val
    trend, color = get_trend(total_score)
    
    return {
        "Get": (get_str, "green" if get_val > 0 else "red"),
        "Set": (set_str, "green" if set_val > 0 else "red"),
        "Go": (go_str, "green" if go_val > 0 else "red"),
        "Trend": (f"{trend} ({total_score})", color)
    }

def main():
    st.title("Stock DMI MACD States Dashboard")
    
    # Sidebar for portfolio selection
    st.sidebar.title("Settings")
    selected_portfolio = st.sidebar.selectbox(
        "Select Portfolio",
        list(default_stocks.keys())
    )
    
    # Get selected symbols
    symbols = default_stocks[selected_portfolio]
    
    # Create empty DataFrame for results
    columns = pd.MultiIndex.from_product([TIMEFRAMES.keys(), ['Get', 'Set', 'Go', 'Trend']])
    results = pd.DataFrame(index=symbols, columns=columns)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Analysis for each symbol and timeframe
    total_iterations = len(symbols) * len(TIMEFRAMES)
    current_iteration = 0
    
    for symbol in symbols:
        status_text.text(f"Analyzing {symbol}...")
        for tf_name, tf_code in TIMEFRAMES.items():
            data = fetch_data(symbol, tf_code)
            analysis = analyze_symbol(data)
            
            if analysis:
                for indicator in ['Get', 'Set', 'Go', 'Trend']:
                    results.loc[symbol, (tf_name, indicator)] = analysis[indicator]
            
            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)
    
    progress_bar.empty()
    status_text.empty()
    
    # Display results in a styled table
    st.markdown("""
    <style>
    .stDataFrame td, .stDataFrame th {
        padding: 5px;
        text-align: center;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Convert results to HTML with colored cells
    html_table = "<table style='width:100%; border-collapse: collapse;'>"
    
    # Header
    html_table += "<tr><th></th>"
    for tf in TIMEFRAMES.keys():
        html_table += f"<th colspan='4' style='text-align:center; border:1px solid gray;'>{tf}</th>"
    html_table += "</tr>"
    
    # Subheader
    html_table += "<tr><th style='border:1px solid gray;'>Symbol</th>"
    for _ in TIMEFRAMES.keys():
        for col in ['Get', 'Set', 'Go', 'Trend']:
            html_table += f"<th style='border:1px solid gray;'>{col}</th>"
    html_table += "</tr>"
    
    # Data rows
    for symbol in symbols:
        html_table += f"<tr><td style='border:1px solid gray;'>{symbol}</td>"
        for tf in TIMEFRAMES.keys():
            for col in ['Get', 'Set', 'Go', 'Trend']:
                value = results.loc[symbol, (tf, col)]
                if isinstance(value, tuple):
                    text, color = value
                    html_table += f"<td style='border:1px solid gray; color:{color};'>{text}</td>"
                else:
                    html_table += f"<td style='border:1px solid gray;'>{value}</td>"
        html_table += "</tr>"
    
    html_table += "</table>"
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Add auto-refresh button
    if st.button("Refresh Data"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()

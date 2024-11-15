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

# [Previous imports and portfolio definitions remain the same]

def calculate_dmi(df, length=14, smoothing=14):
    try:
        df = df.copy()
        
        # Calculate True Range
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        high_diff = df['High'] - df['High'].shift(1)
        low_diff = df['Low'].shift(1) - df['Low']
        
        # Calculate +DM and -DM with proper conditions
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        plus_dm.loc[(high_diff > low_diff) & (high_diff > 0)] = high_diff
        minus_dm.loc[(low_diff > high_diff) & (low_diff > 0)] = low_diff
        
        # Calculate smoothed values using Wilder's smoothing
        def wilder_smooth(series, length):
            # First value is SMA
            smooth = series.rolling(window=length).mean()
            # Calculate subsequent values using Wilder's smoothing
            for i in range(length, len(series)):
                smooth.iloc[i] = (smooth.iloc[i-1] * (length-1) + series.iloc[i]) / length
            return smooth
        
        tr_smooth = wilder_smooth(tr, length)
        plus_dm_smooth = wilder_smooth(plus_dm, length)
        minus_dm_smooth = wilder_smooth(minus_dm, length)
        
        # Calculate +DI and -DI
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, smoothing)
        
        return plus_di, minus_di, adx
        
    except Exception as e:
        st.error(f"Error in DMI calculation: {str(e)}")
        return None, None, None

def get_state(plus_di, minus_di, adx):
    if plus_di is None or minus_di is None or adx is None:
        return 0, "N/A"
    
    try:
        # Check crossovers with proper thresholds
        cross_up = (plus_di.shift(1) <= minus_di.shift(1)) & (plus_di > minus_di)
        cross_down = (plus_di.shift(1) >= minus_di.shift(1)) & (plus_di < minus_di)
        
        # Get last values
        last_plus = plus_di.iloc[-1]
        last_minus = minus_di.iloc[-1]
        last_adx = adx.iloc[-1]
        prev_adx = adx.iloc[-2]
        
        # Strong trend threshold
        strong_trend = last_adx > 25
        
        if cross_up.iloc[-1]:
            return 4, "Bullish++"
        elif cross_down.iloc[-1]:
            return -4, "Bearish++"
        elif last_plus > last_minus:
            if strong_trend and last_adx > prev_adx:
                return 4, "Bullish+"
            else:
                return 3, "Bullish-"
        else:
            if strong_trend and last_adx > prev_adx:
                return -4, "Bearish+"
            else:
                return -3, "Bearish-"
                
    except Exception as e:
        st.error(f"Error in get_state: {str(e)}")
        return 0, "N/A"

def set_state(macd):
    if macd is None:
        return 0, "N/A"
    
    try:
        zero_line = pd.Series(0, index=macd.index)
        
        # Check crossovers with proper thresholds
        cross_up = (macd.shift(1) <= 0) & (macd > 0)
        cross_down = (macd.shift(1) >= 0) & (macd < 0)
        
        # Get momentum
        momentum = macd.diff()
        strong_momentum = abs(momentum.iloc[-1]) > abs(momentum.iloc[-2])
        
        if cross_up.iloc[-1]:
            return 3, "Set Bullish++"
        elif cross_down.iloc[-1]:
            return -3, "Set Bearish++"
        elif macd.iloc[-1] > 0:
            if strong_momentum and macd.iloc[-1] > macd.iloc[-2]:
                return 2, "Bullish+"
            else:
                return 1, "Bullish-"
        else:
            if strong_momentum and macd.iloc[-1] < macd.iloc[-2]:
                return -2, "Bearish+"
            else:
                return -1, "Bearish-"
                
    except Exception as e:
        st.error(f"Error in set_state: {str(e)}")
        return 0, "N/A"

def go_state(signal):
    if signal is None:
        return 0, "N/A"
    
    try:
        zero_line = pd.Series(0, index=signal.index)
        
        # Check crossovers with proper thresholds
        cross_up = (signal.shift(1) <= 0) & (signal > 0)
        cross_down = (signal.shift(1) >= 0) & (signal < 0)
        
        # Get momentum
        momentum = signal.diff()
        strong_momentum = abs(momentum.iloc[-1]) > abs(momentum.iloc[-2])
        
        if cross_up.iloc[-1]:
            return 3, "Go Bullish++"
        elif cross_down.iloc[-1]:
            return -3, "Go Bearish++"
        elif signal.iloc[-1] > 0:
            if strong_momentum and signal.iloc[-1] > signal.iloc[-2]:
                return 2, "Bullish+"
            else:
                return 1, "Bullish-"
        else:
            if strong_momentum and signal.iloc[-1] < signal.iloc[-2]:
                return -2, "Bearish+"
            else:
                return -1, "Bearish-"
                
    except Exception as e:
        st.error(f"Error in go_state: {str(e)}")
        return 0, "N/A"

def get_trend(total_score):
    if abs(total_score) >= 5:
        if total_score > 0:
            return f"Buy ({total_score})", "green"
        else:
            return f"Sell ({total_score})", "red"
    return "", "white"  # Empty string for neutral trend

# [Rest of the code remains the same]


@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_data(symbol, timeframe):
    """
    Fetch data for a given symbol and timeframe with proper handling of Yahoo Finance's new format
    """
    try:
        end_date = datetime.now()
        if timeframe == "1h":
            start_date = end_date - timedelta(days=7)
            interval = "1h"
        elif timeframe == "1d":
            start_date = end_date - timedelta(days=100)
            interval = "1d"
        else:  # Weekly
            start_date = end_date - timedelta(days=365)
            interval = "1wk"
        
        # Create a Ticker object
        ticker = yf.Ticker(symbol)
        
        # Download data
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True
        )
        
        if data.empty:
            return None
            
        if len(data) < 30:
            return None
            
        return data
        
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def analyze_symbol(data):
    if data is None or len(data) < 30:
        return {
            "Get": ("N/A", "white"),
            "Set": ("N/A", "white"),
            "Go": ("N/A", "white"),
            "Trend": ("N/A", "white")
        }
    
    try:
        plus_di, minus_di, adx = calculate_dmi(data)
        macd, signal = calculate_macd(data)
        
        get_val, get_str = get_state(plus_di, minus_di, adx)
        set_val, set_str = set_state(macd)
        go_val, go_str = go_state(signal)
        
        total_score = get_val + set_val + go_val
        trend, color = get_trend(total_score)
        
        return {
            "Get": (get_str, "green" if get_val > 0 else "red" if get_val < 0 else "white"),
            "Set": (set_str, "green" if set_val > 0 else "red" if set_val < 0 else "white"),
            "Go": (go_str, "green" if go_val > 0 else "red" if go_val < 0 else "white"),
            "Trend": (trend, color)
        }
    except Exception as e:
        st.error(f"Error in analysis: {str(e)}")
        return None

def main():
    st.title("Stock DMI MACD States Dashboard")
    
    # Sidebar for portfolio selection
    st.sidebar.title("Settings")
    selected_portfolio = st.sidebar.selectbox(
        "Select Portfolio",
        list(default_stocks.keys())
    )
    
    symbols = default_stocks[selected_portfolio]
    
    # Create empty DataFrame for results
    columns = pd.MultiIndex.from_product([TIMEFRAMES.keys(), ['Get', 'Set', 'Go', 'Trend']])
    results = pd.DataFrame(index=symbols, columns=columns)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_iterations = len(symbols) * len(TIMEFRAMES)
    current_iteration = 0
    
    for symbol in symbols:
        status_text.text(f"Processing {symbol}...")
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
    
    # Table styling
    st.markdown("""
    <style>
    .stDataFrame td, .stDataFrame th {
        padding: 5px;
        text-align: center;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create HTML table
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
    
    # Refresh button
    if st.button("Refresh Data"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()

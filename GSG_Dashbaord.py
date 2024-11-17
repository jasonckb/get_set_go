import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time

# Initialize session state if not already initialized
if 'session_info' not in st.session_state:
    st.session_state.session_info = {}

# Set page config
st.set_page_config(layout="wide", page_title="Stock DMI MACD States Dashboard")

# Define portfolios
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
        "2518.HK", "2628.HK", "3690.HK", "3618.HK", "3888.HK", "3968.HK", "6060.HK", "6078.HK",
        "6098.HK", "6618.HK", "6690.HK", "6862.HK", "9618.HK", "9626.HK", "9698.HK", "9888.HK",
        "9961.HK", "9988.HK", "9999.HK"
    ],
    "US Stocks": ["^NDX", "^SPX"] + [        
        "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLRE", "XLY",
        "AAPL", "ABBV", "ABNB", "ACN", "ADBE", "AMD", "AMGN", "AMZN", "AMT", "ASML",
        "AVGO", "BA", "BKNG", "BLK", "CAT", "CCL", "CDNS", "CEG", "CHTR", "COST", 
        "CRM", "CRWD", "CVS", "CVX", "DDOG", "DE", "DIS", "EQIX", "FTNT", "GE",
        "GILD", "GOOG", "GS", "HD", "IBM", "ICE", "IDXX", "INTC", "INTU", "ISRG",
        "JNJ", "JPM", "KO", "LEN", "LLY", "LRCX", "MA", "META", "MMM", "MRK", 
        "MS", "MSFT", "MU", "NEE", "NFLX", "NRG", "NVO", "NVDA", "OXY", "PANW",
        "PFE", "PG", "PGR", "PLTR", "PYPL", "QCOM", "REGN", "SBUX", "SMH", "SNOW",
        "SPGI", "TEAM", "TJX", "TRAV", "TSM", "TSLA", "TTD", "TXN", "UNH", "UPS",
        "V", "VST", "VZ", "WMT", "XOM", "ZS"
    ],
    "World Index": [
        "^SPX", "^NDX", "^RUT", "^SOX", "^TNX", "^DJI", "^HSI", "3032.HK", 
        "^N225", "^BSESN", "^KS11", "^TWII", "^GDAXI", "^FTSE", "^FCHI", "^BVSP", "EEMA", 
        "EEM", "^HUI", "CL=F", "GC=F", "HG=F", "SI=F", "DX-Y.NYB", "BTC=F", "ETH=F"
    ]
}

TIMEFRAMES = {
    "Weekly": "1wk",
    "Daily": "1d",
    "Hourly": "1h"
}

def rma(series, length):
    """Replicate TradingView's ta.rma function exactly"""
    alpha = 1.0 / length
    result = pd.Series(0.0, index=series.index)
    
    # Find first valid value
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        return result
    
    # Set first value
    result.loc[first_valid_idx] = series.loc[first_valid_idx]
    
    # Calculate RMA
    for i in range(series.index.get_loc(first_valid_idx) + 1, len(series)):
        if pd.isna(series.iloc[i]):
            result.iloc[i] = result.iloc[i-1]
        else:
            result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
    
    return result

def calculate_dmi(df, length=14, smoothing=14):
    try:
        df = df.copy()
        
        # Calculate directional movement exactly like TradingView
        up = df['High'] - df['High'].shift(1)  # ta.change(high)
        down = -(df['Low'] - df['Low'].shift(1))  # -ta.change(low)
        
        # Calculate DM exactly like TradingView
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        plus_dm[(up > down) & (up > 0)] = up
        minus_dm[(down > up) & (down > 0)] = down
        
        # Calculate True Range
        tr = pd.DataFrame({
            'hl': df['High'] - df['Low'],
            'hc': abs(df['High'] - df['Close'].shift(1)),
            'lc': abs(df['Low'] - df['Close'].shift(1))
        }).max(axis=1)
        
        # Use RMA for smoothing
        tr_rma = rma(tr, length)
        plus_di = 100 * rma(plus_dm, length) / tr_rma
        minus_di = 100 * rma(minus_dm, length) / tr_rma
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
        adx = rma(dx, smoothing)
        
        return plus_di, minus_di, adx
        
    except Exception as e:
        st.error(f"Error in DMI calculation: {str(e)}")
        return None, None, None

def pine_ema(series, length, alpha_adj=19):
    """Replicate TradingView's pine_ema function exactly"""
    alpha = 2.0 / (length + alpha_adj)
    result = pd.Series(index=series.index)
    
    # Initialize with NaN
    result.iloc[0] = series.iloc[0]
    
    # Calculate exactly like TradingView:
    # sum := na(sum[1]) ? src : alpha * src + (1 - alpha) * nz(sum[1])
    for i in range(1, len(series)):
        prev_sum = result.iloc[i-1]
        curr_src = series.iloc[i]
        
        if pd.isna(prev_sum):
            # If previous sum is NA, use current source value
            result.iloc[i] = curr_src
        else:
            # Otherwise use EMA formula
            result.iloc[i] = alpha * curr_src + (1 - alpha) * (prev_sum if not pd.isna(prev_sum) else 0)
    
    return result

def calculate_macd(df, fast_length=12, slow_length=26, signal_length=9, alpha_adj=19):
    try:
        # Use Close price
        close = df['Close'].copy()
        
        # Calculate MACD using pine_ema exactly like TradingView
        fast_ma = pine_ema(close, fast_length, alpha_adj)
        slow_ma = pine_ema(close, slow_length, alpha_adj)
        macd = fast_ma - slow_ma
        signal = pine_ema(macd, signal_length, alpha_adj)
        
        return macd, signal
        
    except Exception as e:
        st.error(f"Error in MACD calculation: {str(e)}")
        return None, None

def get_state(plus_di, minus_di, adx):
    if plus_di is None or minus_di is None or adx is None:
        return 0, "N/A"
    
    try:
        # Check crossovers
        dmi_cross_up = (plus_di.shift(1) <= minus_di.shift(1)) & (plus_di > minus_di)
        dmi_cross_down = (plus_di.shift(1) >= minus_di.shift(1)) & (plus_di < minus_di)
        
        if dmi_cross_up.iloc[-1]:
            return 4, "Bullish++"
        elif dmi_cross_down.iloc[-1]:
            return -4, "Bearish++"
        elif plus_di.iloc[-1] > minus_di.iloc[-1]:
            if adx.iloc[-1] > adx.iloc[-2]:
                return 4, "Bullish+"
            else:
                return 3, "Bullish-"
        else:
            if adx.iloc[-1] > adx.iloc[-2]:
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
        cross_up = (macd.shift(1) <= 0) & (macd > 0)
        cross_down = (macd.shift(1) >= 0) & (macd < 0)
        
        if cross_up.iloc[-1]:
            return 2, "Set Bullish++"
        elif cross_down.iloc[-1]:
            return -2, "Set Bearish++"
        elif macd.iloc[-1] > 0:
            if macd.iloc[-1] > macd.iloc[-2]:
                return 2, "Bullish+"
            else:
                return 1, "Bullish-"
        else:
            if macd.iloc[-1] < macd.iloc[-2]:
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
        cross_up = (signal.shift(1) <= 0) & (signal > 0)
        cross_down = (signal.shift(1) >= 0) & (signal < 0)
        
        if cross_up.iloc[-1]:
            return 2, "Go Bullish++"
        elif cross_down.iloc[-1]:
            return -2, "Go Bearish++"
        elif signal.iloc[-1] > 0:
            if signal.iloc[-1] > signal.iloc[-2]:
                return 2, "Bullish+"
            else:
                return 1, "Bullish-"
        else:
            if signal.iloc[-1] < signal.iloc[-2]:
                return -2, "Bearish+"
            else:
                return -1, "Bearish-"
                
    except Exception as e:
        st.error(f"Error in go_state: {str(e)}")
        return 0, "N/A"

def get_trend(total_score):
    if total_score >= 5:
        return f"Buy ({total_score})", "green"
    elif total_score <= -5:
        return f"Sell ({total_score})", "red"
    else:
        return f"Hold ({total_score})", "gray"

@st.cache_data(ttl=300)  # Cache data for 5 minutes
def fetch_data(symbol, timeframe):
    try:
        end_date = datetime.now()
        ticker = yf.Ticker(symbol)
        
        if timeframe == "1h":
            start_date = end_date - timedelta(days=20)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval="1h",
                auto_adjust=True
            )
        elif timeframe == "1d":
            start_date = end_date - timedelta(days=250)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True
            )
        else:  # Weekly
            start_date = end_date - timedelta(days=1000)
            # Get daily data first
            daily_data = ticker.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True
            )
            
            # Check if it's a HK stock
            is_hk_stock = symbol.endswith('.HK')
            
            # Define resampling functions
            functions = {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            }
            
            if is_hk_stock:
                # For HK stocks, handle timezone appropriately
                daily_data.index = daily_data.index.tz_localize(None)
                daily_data.index = daily_data.index.tz_localize('Asia/Hong_Kong')
                data = daily_data.resample('W-FRI', closed='right', label='right').agg(functions)
                data.index = data.index.tz_localize(None)
            else:
                # For other stocks, use standard resampling
                data = daily_data.resample('W-FRI').agg(functions)
        
        if data.empty:
            return None
            
        if len(data) < 30:
            return None
        
        # Fill any missing data
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure index has no timezone info
        data.index = data.index.tz_localize(None)
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
        # Calculate indicators
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

def extract_trend_value(trend_str):
    """Extract numeric value from trend string."""
    if not isinstance(trend_str, tuple):
        return 0
    trend_text = trend_str[0]
    if not isinstance(trend_text, str):
        return 0
    import re
    match = re.search(r'\(([+-]?\d+)\)', trend_text)
    return int(match.group(1)) if match else 0

def calculate_total_trend(weekly_trend, daily_trend, hourly_trend):
    """Calculate weighted average trend."""
    weekly_val = extract_trend_value(weekly_trend)
    daily_val = extract_trend_value(daily_trend)
    hourly_val = extract_trend_value(hourly_trend)
    
    weighted_sum = (weekly_val * 2 + daily_val * 2 + hourly_val * 1) / 5
    
    if weighted_sum >= 5:
        return (f"Buy ({weighted_sum:.1f})", "green")
    elif weighted_sum <= -5:
        return (f"Sell ({weighted_sum:.1f})", "red")
    else:
        return (f"Hold ({weighted_sum:.1f})", "gray")

def main():
    st.title("Get Set Go Dashboard")
    
    # Clear cache if refresh button is clicked
    if st.button("Refresh Data"):
        st.cache_data.clear()
    
    # Sidebar for portfolio selection
    st.sidebar.title("Settings")
    selected_portfolio = st.sidebar.selectbox(
        "Select Portfolio",
        options=list(default_stocks.keys()),
        key="portfolio_selector"
    )
    
    # Get selected symbols
    symbols = default_stocks[selected_portfolio]
    
    # Store data and calculations for debug view
    debug_data = {}
    
    # Create empty DataFrame for results
    # Modified to include Total Trend column
    timeframe_columns = ['Get', 'Set', 'Go', 'Trend']
    columns = ['Total Trend'] + [f"{tf}_{col}" for tf in TIMEFRAMES.keys() for col in timeframe_columns]
    results = pd.DataFrame(index=symbols, columns=columns)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_iterations = len(symbols) * len(TIMEFRAMES)
    current_iteration = 0
    
    # Store last update time for each timeframe
    last_update_times = {}
    
    # Process each symbol
    for symbol in symbols:
        status_text.text(f"Processing {symbol}...")
        debug_data[symbol] = {}
        
        timeframe_results = {}
        
        # Process each timeframe
        for tf_name, tf_code in TIMEFRAMES.items():
            data = fetch_data(symbol, tf_code)
            if data is not None:
                # Store the last update time for each timeframe
                if tf_name not in last_update_times:
                    last_update_times[tf_name] = data.index[-1]
                
                # Store raw data
                debug_data[symbol][tf_name] = {
                    'raw_data': data.tail(),
                    'calculations': {}
                }
                
                # Calculate indicators
                plus_di, minus_di, adx = calculate_dmi(data)
                macd, signal = calculate_macd(data)
                
                # Store calculations
                debug_data[symbol][tf_name]['calculations'] = {
                    'plus_di': plus_di.tail() if plus_di is not None else None,
                    'minus_di': minus_di.tail() if minus_di is not None else None,
                    'adx': adx.tail() if adx is not None else None,
                    'macd': macd.tail() if macd is not None else None,
                    'signal': signal.tail() if signal is not None else None
                }
            
            analysis = analyze_symbol(data)
            
            if analysis:
                # Store results for each timeframe
                timeframe_results[tf_name] = analysis
                for indicator in ['Get', 'Set', 'Go', 'Trend']:
                    results.loc[symbol, f"{tf_name}_{indicator}"] = analysis[indicator]
            
            current_iteration += 1
            progress_bar.progress(current_iteration / total_iterations)
        
        # Calculate Total Trend
        if timeframe_results:
            total_trend = calculate_total_trend(
                timeframe_results['Weekly']['Trend'],
                timeframe_results['Daily']['Trend'],
                timeframe_results['Hourly']['Trend']
            )
            results.loc[symbol, 'Total Trend'] = total_trend
    
    progress_bar.empty()
    status_text.empty()
    
    # Table styling (updated to include Total Trend)
    st.markdown("""
    <style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        border: 1px solid gray;
        padding: 8px;
        text-align: left;
    }
    .timeframe {
        text-align: center;
        font-weight: bold;
    }
    .symbol {
        text-align: left;
    }
    .value {
        text-align: center;
    }
    .last-update {
        text-align: center;
        font-style: italic;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create HTML table with Total Trend
    html_table = "<table>"
    
    # Last update time row
    html_table += "<tr><th></th><th></th>"  # Extra th for Total Trend
    for tf in TIMEFRAMES.keys():
        last_update = last_update_times.get(tf, "N/A")
        if isinstance(last_update, pd.Timestamp):
            last_update_str = last_update.strftime("%Y-%m-%d %H:%M:%S")
        else:
            last_update_str = str(last_update)
        html_table += f"<th colspan='4' class='last-update'>Last Update: {last_update_str}</th>"
    html_table += "</tr>"
    
    # Header
    html_table += "<tr><th></th><th class='timeframe'>Total Trend</th>"
    for tf in TIMEFRAMES.keys():
        html_table += f"<th colspan='4' class='timeframe'>{tf}</th>"
    html_table += "</tr>"
    
    # Subheader
    html_table += "<tr><th class='symbol'>Symbol</th><th class='value'></th>"
    for _ in TIMEFRAMES.keys():
        html_table += "<th class='value'>Get</th><th class='value'>Set</th><th class='value'>Go</th><th class='value'>Trend</th>"
    html_table += "</tr>"
    
    # Data rows
    for symbol in symbols:
        html_table += f"<tr><td class='symbol'>{symbol}</td>"
        
        # Add Total Trend
        total_trend = results.loc[symbol, 'Total Trend']
        if isinstance(total_trend, tuple):
            text, color = total_trend
            html_table += f"<td class='value' style='color:{color};'>{text}</td>"
        else:
            html_table += f"<td class='value'>{total_trend}</td>"
        
        # Add other columns
        for tf in TIMEFRAMES.keys():
            for col in ['Get', 'Set', 'Go', 'Trend']:
                value = results.loc[symbol, f"{tf}_{col}"]
                if isinstance(value, tuple):
                    text, color = value
                    html_table += f"<td class='value' style='color:{color};'>{text}</td>"
                else:
                    html_table += f"<td class='value'>{value}</td>"
        html_table += "</tr>"
    
    html_table += "</table>"
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Debug section remains the same...

# [Previous code remains the same until fetch_data function]

def fetch_data(symbol, timeframe):
    """
    Fetch data for a given symbol and timeframe with proper handling of Yahoo Finance's new format
    """
    try:
        # Show fetching status
        status_container = st.empty()
        status_container.info(f"Fetching {timeframe} data for {symbol}...")
        
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
        
        # Add retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Download data
                data = yf.download(
                    symbol,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    show_errors=True,
                    ignore_tz=True
                )
                
                # Handle the new Yahoo Finance format
                if isinstance(data.index[0], str):  # Check if first row is ticker name
                    # Skip the ticker name row and convert remaining data
                    data = data.iloc[1:]  # Skip first row
                    data.index = pd.to_datetime(data.index)  # Convert index to datetime
                
                if data.empty:
                    status_container.warning(f"No data available for {symbol} on {timeframe} timeframe")
                    return None
                
                if len(data) < 30:
                    status_container.warning(f"Insufficient data points for {symbol} on {timeframe} timeframe (got {len(data)}, need at least 30)")
                    return None
                
                # Ensure all required columns exist
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_columns):
                    status_container.error(f"Missing required columns for {symbol}")
                    return None
                
                # Ensure data types are correct
                for col in ['Open', 'High', 'Low', 'Close']:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                
                # Drop any rows with NaN values
                data = data.dropna()
                
                if len(data) < 30:
                    status_container.warning(f"Insufficient valid data points after cleaning for {symbol}")
                    return None
                
                # Show success message and clear container
                status_container.success(f"Successfully fetched {len(data)} data points for {symbol}")
                time.sleep(0.5)  # Show success message briefly
                status_container.empty()
                return data
                
            except Exception as e:
                if attempt < max_retries - 1:
                    status_container.warning(f"Attempt {attempt + 1} failed for {symbol}. Retrying...")
                    time.sleep(1)  # Wait before retrying
                else:
                    status_container.error(f"Failed to fetch data for {symbol} after {max_retries} attempts: {str(e)}")
                    return None
                
    except Exception as e:
        st.error(f"Error in fetch_data for {symbol}: {str(e)}")
        return None

# [Rest of the code remains the same]

import yfinance as yf
import pandas as pd
import pandas_ta as ta

# ==========================================
# 1. SETTINGS
# ==========================================
TICKER = "^NSEI"      # Nifty 50 Index
INTERVAL = "15m"      # 15 Minute Timeframe
PERIOD = "60d"        # Last 60 Days
EMA_LEN = 20
ADX_LEN = 14
ADX_THRESHOLD = 20    # Trend Strength Filter
BROKERAGE_PER_TRADE = 50 
INITIAL_CAPITAL = 100000

print(f"Fetching data for {TICKER}...")
df = yf.download(TICKER, period=PERIOD, interval=INTERVAL, progress=False)

# Clean up data structure
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# --- THE FIX: CONVERT TIMEZONE TO IST ---
# Check if data has timezone info, if not assume UTC then convert
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')
df.index = df.index.tz_convert('Asia/Kolkata')

print("Data converted to IST (Indian Standard Time).")

# ==========================================
# 2. CALCULATE INDICATORS
# ==========================================
df['EMA_High'] = ta.ema(df['High'], length=EMA_LEN)
df['EMA_Low'] = ta.ema(df['Low'], length=EMA_LEN)
adx_df = ta.adx(df['High'], df['Low'], df['Close'], length=ADX_LEN)
df['ADX'] = adx_df[f'ADX_{ADX_LEN}']

# ==========================================
# 3. STRATEGY LOGIC
# ==========================================
trades = []
position = None 
entry_price = 0
entry_time = None

print("Running Backtest Logic on IST Data...")

for i in range(1, len(df)):
    curr = df.iloc[i]
    prev = df.iloc[i-1]
    
    # Time Filter: Trade after 09:30 AM IST
    is_time_valid = (curr.name.hour > 9) or (curr.name.hour == 9 and curr.name.minute >= 30)
    
    # Stop trading after 03:00 PM (Avoid end of day volatility)
    is_before_close = (curr.name.hour < 15)
    
    is_trend_strong = curr['ADX'] > ADX_THRESHOLD
    
    # ----------------------------------
    # ENTRY LOGIC
    # ----------------------------------
    if position is None and is_time_valid and is_before_close and is_trend_strong:
        
        # BUY: Crossover EMA High
        if prev['Close'] < prev['EMA_High'] and curr['Close'] > curr['EMA_High']:
            position = "LONG"
            entry_price = curr['Close']
            entry_time = curr.name
            
        # SELL: Crossunder EMA Low
        elif prev['Close'] > prev['EMA_Low'] and curr['Close'] < curr['EMA_Low']:
            position = "SHORT"
            entry_price = curr['Close']
            entry_time = curr.name

    # ----------------------------------
    # EXIT LOGIC
    # ----------------------------------
    elif position == "LONG":
        # Exit if Price Crosses UNDER EMA Low
        if curr['Close'] < curr['EMA_Low']:
            pnl = curr['Close'] - entry_price
            trades.append({'Type': 'Long', 'Entry': entry_price, 'Exit': curr['Close'], 'PnL': pnl, 'Time': entry_time})
            position = None

    elif position == "SHORT":
        # Exit if Price Crosses OVER EMA High
        if curr['Close'] > curr['EMA_High']:
            pnl = entry_price - curr['Close']
            trades.append({'Type': 'Short', 'Entry': entry_price, 'Exit': curr['Close'], 'PnL': pnl, 'Time': entry_time})
            position = None

# ==========================================
# 4. RESULTS
# ==========================================
if len(trades) >
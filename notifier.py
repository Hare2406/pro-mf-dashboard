import os
import requests
import pandas as pd
import json
from mftool import Mftool
from ta.momentum import RSIIndicator

def send_telegram(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

def run_check():
    obj = Mftool()
    
    # Logic to find your existing watchlist JSON file
    watchlist_file = "mf_watchlist_data.json"
    if os.path.exists(watchlist_file):
        with open(watchlist_file, 'r') as f:
            codes = json.load(f)
    else:
        # Fallback codes if file is not found (SBI Small Cap, Quant)
        codes = ["125497", "118989"] 

    report = "🚀 *Wealth Alert: Daily RSI Update*\n"
    
    for code in codes:
        try:
            data = obj.get_scheme_historical_nav(code, as_json=False)
            df = pd.DataFrame(data['data'])
            df['nav'] = pd.to_numeric(df['nav'])
            df = df.iloc[::-1] # Ensure chronological order
            
            rsi = RSIIndicator(close=df['nav'], window=14).rsi().iloc[-1]
            name = data['meta']['scheme_name'][:20]
            
            # Formatting the Alert
            icon = "🔴 BUY" if rsi < 35 else "🟢 OK"
            report += f"\n{icon} | {name}\n   RSI: *{rsi:.1f}*"
        except Exception:
            continue

    send_telegram(report)

if __name__ == "__main__":
    run_check()

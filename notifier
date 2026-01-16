import os
import requests
import pandas as pd
from mftool import Mftool
from ta.momentum import RSIIndicator

def send_whatsapp(message):
    api_key = os.getenv("WHATSAPP_API_KEY")
    phone = os.getenv("WHATSAPP_PHONE")
    # URL Encoding spaces and new lines
    message = message.replace(" ", "+").replace("\n", "%0A")
    url = f"https://api.callmebot.com/whatsapp.php?phone={phone}&text={message}&apikey={api_key}"
    requests.get(url)

def run_check():
    obj = Mftool()
    # Replace these codes with your actual watchlist codes
    watchlist = ["125497", "118989", "101237"] 
    report = "📈 *Daily MF RSI Report*\n"
    
    for code in watchlist:
        try:
            data = obj.get_scheme_historical_nav(code, as_json=False)
            df = pd.DataFrame(data['data'])
            df['nav'] = pd.to_numeric(df['nav'])
            df = df.iloc[::-1] # Newest last
            
            rsi = RSIIndicator(close=df['nav'], window=14).rsi().iloc[-1]
            name = data['meta']['scheme_name'][:20]
            
            icon = "🔴" if rsi < 35 else "🟢"
            report += f"\n{icon} {name}: *{rsi:.1f}*"
        except:
            continue

    send_whatsapp(report)

if __name__ == "__main__":
    run_check()

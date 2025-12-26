import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mftool import Mftool
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta
from scipy import optimize
import os
import json
import concurrent.futures # Speed Boost

# --- Configuration ---
st.set_page_config(page_title="Pro MF Dashboard", layout="wide", page_icon="📈")
obj = Mftool()

# --- CONSTANTS ---
USER_HOME = os.path.expanduser("~")
WATCHLIST_FILE = os.path.join(USER_HOME, "mf_watchlist_data.json")

# --- Initialize Session State ---
if 'selected_scheme_code' not in st.session_state:
    st.session_state.selected_scheme_code = None 

# --- Storage Functions ---
def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        try:
            with open(WATCHLIST_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_watchlist(codes):
    try:
        with open(WATCHLIST_FILE, 'w') as f:
            json.dump(codes, f)
    except Exception as e:
        st.toast(f"⚠️ Save Warning: {e}")

# --- Data Engine (Optimized) ---
@st.cache_data(ttl=24*3600)
def get_all_schemes(): 
    return obj.get_scheme_codes()

@st.cache_data(ttl=3600)
def get_fund_data(code):
    try:
        data = obj.get_scheme_historical_nav(code, as_json=False)
        if data and 'data' in data and len(data['data']) > 0:
            df = pd.DataFrame(data['data'])
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            df['nav'] = pd.to_numeric(df['nav'])
            df = df.sort_values('date')
            return df
        return pd.DataFrame()
    except: 
        return pd.DataFrame()

def fetch_data_parallel(codes_list):
    """Fetches data for multiple funds at once (Speed Boost)"""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_code = {executor.submit(get_fund_data, code): code for code in codes_list}
        for future in concurrent.futures.as_completed(future_to_code):
            code = future_to_code[future]
            try:
                data = future.result()
                results[code] = data
            except:
                results[code] = pd.DataFrame()
    return results

# --- Calculation Helpers ---
def calculate_indicators(df):
    if df.empty or len(df) < 50: return None
    
    df_calc = df.copy()
    if 'date' in df_calc.columns:
        df_calc.set_index('date', inplace=True)
    
    # 1. Daily Indicators
    rsi_d_14 = RSIIndicator(close=df_calc['nav'], window=14).rsi().iloc[-1]
    rsi_d_2 = RSIIndicator(close=df_calc['nav'], window=2).rsi().iloc[-1]
    
    ema_20 = EMAIndicator(close=df_calc['nav'], window=20).ema_indicator().iloc[-1]
    ema_50 = EMAIndicator(close=df_calc['nav'], window=50).ema_indicator().iloc[-1]
    ema_signal = "🟢 Bullish" if ema_20 > ema_50 else "🔴 Bearish"

    # 2. Weekly Indicators
    df_w = df_calc.resample('W-FRI').last().dropna()
    
    if len(df_w) > 14:
        rsi_w_14 = RSIIndicator(close=df_w['nav'], window=14).rsi().iloc[-1]
        rsi_w_2 = RSIIndicator(close=df_w['nav'], window=2).rsi().iloc[-1]
    else:
        rsi_w_14 = 0; rsi_w_2 = 0

    return {
        "RSI_D_14": round(rsi_d_14, 1),
        "RSI_D_2": round(rsi_d_2, 1),
        "RSI_W_14": round(rsi_w_14, 1),
        "RSI_W_2": round(rsi_w_2, 1),
        "EMA_Signal": ema_signal,
        "NAV": df['nav'].iloc[-1],
        "Date": df['date'].iloc[-1].strftime('%d-%b')
    }

def xirr(transactions):
    dates = [t[0] for t in transactions]
    amounts = [t[1] for t in transactions]
    if not amounts or sum(amounts) == 0: return 0.0
    def xnpv(rate, amounts, dates):
        if rate <= -1.0: return float('inf')
        d0 = dates[0]
        return sum([a / (1 + rate)**((d - d0).days / 365.0) for a, d in zip(amounts, dates)])
    try:
        return optimize.newton(lambda r: xnpv(r, amounts, dates), 0.1)
    except:
        return 0.0

# --- 1. Sidebar ---
st.sidebar.header("🔍 Fund Selector")

schemes_dict = get_all_schemes()
# Create Map: Name -> Code
name_to_code = {f"{v}": k for k, v in schemes_dict.items()}
scheme_names = list(name_to_code.keys())

# Default Fund Handling
default_fund_code = "125497" # SBI Small Cap
if st.session_state.selected_scheme_code is None:
    st.session_state.selected_scheme_code = default_fund_code

# Find Index for Selectbox based on Session State
try:
    # Reverse lookup to find name from code
    current_name = [name for name, code in name_to_code.items() if code == st.session_state.selected_scheme_code][0]
    default_index = scheme_names.index(current_name)
except:
    default_index = 0

# Selectbox showing ONLY Names
selected_scheme_name = st.sidebar.selectbox("Search Fund", scheme_names, index=default_index)

# Update Session State based on Sidebar Selection
st.session_state.selected_scheme_code = name_to_code[selected_scheme_name]
selected_scheme_code = st.session_state.selected_scheme_code

# --- 2. Data Engine ---
with st.spinner("Processing data..."):
    df_raw = get_fund_data(selected_scheme_code)

# --- TABS ---
tab_chart, tab_strategy, tab_watchlist = st.tabs(["📊 Live Chart", "🧪 Smart Strategy", "📋 Watchlist"])

# ==========================================
# TAB 1: LIVE CHART (Updated with Name & 4 EMAs)
# ==========================================
with tab_chart:
    if df_raw.empty: st.error("Data Unavailable."); st.stop()
    
    # 1. SHOW FUND NAME AT TOP
    st.header(selected_scheme_name)
    st.caption(f"Fund Code: {selected_scheme_code} | Latest NAV: ₹{df_raw['nav'].iloc[-1]}")
    
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Chart Settings")
    chart_height = st.sidebar.slider("📐 Chart Height", 500, 1500, 700, 50)
    
    interval_options = { "Daily": "D", "Weekly (Fri)": "W-FRI", "2 Weeks": "2W-FRI", "Monthly": "ME" }
    selected_interval_label = st.sidebar.selectbox("📅 Interval", list(interval_options.keys()), index=0)
    resample_freq = interval_options[selected_interval_label]

    # 2. UPDATED EMA SETTINGS (4 EMAs)
    with st.sidebar.expander("EMA Settings", expanded=True):
        st.caption("Toggle indicators on/off:")
        
        c_ema1, c_ema2 = st.columns(2)
        show_ema1 = c_ema1.checkbox("EMA 1", True)
        ema1_len = c_ema2.number_input("Len 1", value=20)
        
        c_ema3, c_ema4 = st.columns(2)
        show_ema2 = c_ema3.checkbox("EMA 2", True)
        ema2_len = c_ema4.number_input("Len 2", value=50)

        c_ema5, c_ema6 = st.columns(2)
        show_ema3 = c_ema5.checkbox("EMA 3", False) # Default OFF
        ema3_len = c_ema6.number_input("Len 3", value=100)
        
        c_ema7, c_ema8 = st.columns(2)
        show_ema4 = c_ema7.checkbox("EMA 4", False) # Default OFF
        ema4_len = c_ema8.number_input("Len 4", value=200)

    with st.sidebar.expander("RSI Settings"):
        rsi_len_chart = st.number_input("RSI Length (Chart)", value=14)
        show_rsi_smooth = st.checkbox("Show Smoothed RSI", True)
        rsi_smooth_len = st.number_input("Smoothing Length", value=14)
        
    timeframe_map = { "6M": 180, "1Y": 365, "3Y": 1095, "5Y": 1825, "All": 99999 }
    timeframe = st.sidebar.radio("Lookback:", list(timeframe_map.keys()), horizontal=True, index=2)

    df_chart = df_raw.copy()
    df_chart.set_index('date', inplace=True)
    if resample_freq != "D": df_chart = df_chart.resample(resample_freq).last().dropna()
    df_chart.reset_index(inplace=True)

    df_chart['RSI'] = RSIIndicator(close=df_chart['nav'], window=rsi_len_chart).rsi()
    if show_rsi_smooth: df_chart['RSI_Smooth'] = EMAIndicator(close=df_chart['RSI'], window=rsi_smooth_len).ema_indicator()
    
    # Calculate Active EMAs
    if show_ema1: df_chart[f'EMA_{ema1_len}'] = EMAIndicator(close=df_chart['nav'], window=ema1_len).ema_indicator()
    if show_ema2: df_chart[f'EMA_{ema2_len}'] = EMAIndicator(close=df_chart['nav'], window=ema2_len).ema_indicator()
    if show_ema3: df_chart[f'EMA_{ema3_len}'] = EMAIndicator(close=df_chart['nav'], window=ema3_len).ema_indicator()
    if show_ema4: df_chart[f'EMA_{ema4_len}'] = EMAIndicator(close=df_chart['nav'], window=ema4_len).ema_indicator()

    days = timeframe_map[timeframe]
    start_date = df_chart['date'].iloc[-1] - timedelta(days=days)
    df_filtered = df_chart.loc[df_chart['date'] >= start_date].copy()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], subplot_titles=("Price", "RSI"), vertical_spacing=0.03)
    fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['nav'], name='NAV', line=dict(color='#2962FF', width=2)), row=1, col=1)
    
    # Add EMAs with Distinct Colors
    if show_ema1: fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered[f'EMA_{ema1_len}'], line=dict(color='red', width=1), name=f'EMA {ema1_len}'), row=1, col=1)
    if show_ema2: fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered[f'EMA_{ema2_len}'], line=dict(color='green', width=1), name=f'EMA {ema2_len}'), row=1, col=1)
    if show_ema3: fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered[f'EMA_{ema3_len}'], line=dict(color='orange', width=1), name=f'EMA {ema3_len}'), row=1, col=1)
    if show_ema4: fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered[f'EMA_{ema4_len}'], line=dict(color='lavender', width=1), name=f'EMA {ema4_len}'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['RSI'], name='RSI', line=dict(color='#7E57C2')), row=2, col=1)
    if show_rsi_smooth: fig.add_trace(go.Scatter(x=df_filtered['date'], y=df_filtered['RSI_Smooth'], name='Signal', line=dict(color='yellow')), row=2, col=1)
    
    fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="gray")
    fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="gray")
    fig.add_hrect(y0=30, y1=70, fillcolor="gray", opacity=0.1, line_width=0, row=2, col=1)
    
    config = {'scrollZoom': True, 'displayModeBar': False}
    fig.update_layout(template="plotly_dark", height=chart_height, hovermode="x unified", dragmode="pan", xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True, config=config)

# ==========================================
# TAB 2: SMART STRATEGY (Same Logic)
# ==========================================
with tab_strategy:
    if df_raw.empty: st.error("Data Unavailable."); st.stop()
    st.header(f"🧪 Smart SIP: {selected_scheme_name}")
    st.caption("Strategy: Invest Base SIP every month. If RSI is low, add Top-Up amount.")

    c1, c2, c3, c4 = st.columns(4)
    sip_amount = c1.number_input("Base SIP (₹)", value=6000)
    topup_amount = c2.number_input("Extra Top-Up (₹)", value=500)
    check_mode = c3.selectbox("RSI Logic", ["Daily", "Weekly"], index=0)
    strat_rsi_len = c4.number_input("RSI Length", value=2)
    c5, c6 = st.columns(2)
    strat_rsi_limit = c5.number_input("Buy Threshold (<)", value=30)
    start_year = c6.selectbox("Start Year", range(2010, 2026), index=5)
    sip_day = 5

    if st.button("🚀 Run Backtest", key="run_strat"):
        df_sim = df_raw.copy()
        df_sim['date'] = pd.to_datetime(df_sim['date']); df_sim.sort_values('date', inplace=True); df_sim.set_index('date', inplace=True)
        if check_mode == "Weekly":
            df_w = df_sim.resample('W-FRI').last().dropna(); df_w['RSI_Strat'] = RSIIndicator(close=df_w['nav'], window=strat_rsi_len).rsi()
            df_sim = df_sim.reset_index(); df_w = df_w.reset_index()[['date', 'RSI_Strat']]; df_test = pd.merge_asof(df_sim, df_w, on='date', direction='backward')
        else:
            df_sim['RSI_Strat'] = RSIIndicator(close=df_sim['nav'], window=strat_rsi_len).rsi(); df_test = df_sim.reset_index()

        df_test = df_test[df_test['date'].dt.year >= start_year].reset_index(drop=True)
        if df_test.empty: st.error("No data."); st.stop()

        reg_units = 0; reg_invested = 0; reg_transactions = []
        smart_units = 0; smart_invested = 0; smart_transactions = []
        smart_log = []
        
        df_test['YYYYMM'] = df_test['date'].dt.strftime('%Y%m')
        unique_months = df_test['YYYYMM'].unique()
        
        for month_str in unique_months:
            mask = (df_test['YYYYMM'] == month_str) & (df_test['date'].dt.day >= sip_day)
            month_data = df_test[mask]
            if not month_data.empty:
                row = month_data.iloc[0]; curr = row['date']; nav = row['nav']; rsi_val = row['RSI_Strat']
                reg_units += sip_amount / nav; reg_invested += sip_amount; reg_transactions.append((curr, -sip_amount))
                invest_now = sip_amount
                if rsi_val < strat_rsi_limit:
                    invest_now += topup_amount; smart_log.append(f"⚡ Boosted (+₹{topup_amount}): {curr.strftime('%d-%b-%Y')} | {check_mode} RSI: {rsi_val:.1f}")
                smart_units += invest_now / nav; smart_invested += invest_now; smart_transactions.append((curr, -invest_now))

        curr_nav = df_test['nav'].iloc[-1]
        r_tot = reg_units * curr_nav; reg_transactions.append((df_test['date'].iloc[-1], r_tot)); r_xirr = xirr(reg_transactions) * 100
        s_tot = smart_units * curr_nav; smart_transactions.append((df_test['date'].iloc[-1], s_tot)); s_xirr = xirr(smart_transactions) * 100

        st.table(pd.DataFrame({
            "Metric": ["Total Invested", "Profit", "Total Value", "XIRR"],
            "Regular SIP": [f"₹{reg_invested:,.0f}", f"₹{r_tot - reg_invested:,.0f}", f"₹{r_tot:,.0f}", f"{r_xirr:.2f}%"],
            "Smart SIP": [f"₹{smart_invested:,.0f}", f"₹{s_tot - smart_invested:,.0f}", f"₹{s_tot:,.0f}", f"{s_xirr:.2f}%"]
        }))
        with st.expander(f"View Boosted Months ({len(smart_log)})"):
            for l in smart_log: st.write(l)

# ==========================================
# TAB 3: WATCHLIST (Updated & Speed Boosted)
# ==========================================
with tab_watchlist:
    st.header("📋 Fund Watchlist")
    
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = load_watchlist()

    # Add Fund Section
    c_add1, c_add2 = st.columns([3, 1])
    with c_add1:
        # Show Names only in Selectbox
        wl_fund_name = st.selectbox("Select to Add", scheme_names, key='wl_sel')
    with c_add2:
        if st.button("➕ Add Fund"):
            code = name_to_code[wl_fund_name]
            if code not in st.session_state.watchlist:
                st.session_state.watchlist.append(code)
                save_watchlist(st.session_state.watchlist) 
                st.success("Added!"); st.rerun()
            else:
                st.warning("Exists!")

    st.markdown("---")
    
    # Headers
    h1, h2, h3, h4, h5, h6 = st.columns([0.5, 3, 1, 1, 1, 0.5])
    h1.markdown("**Chart**")
    h2.markdown("**Fund Name**")
    h3.markdown("**NAV**")
    h4.markdown("**Daily RSI**")
    h5.markdown("**Trend (EMA)**")
    h6.markdown("**Del**")
    st.divider()

    # Display Rows
    if st.session_state.watchlist:
        
        # SPEED BOOST: Load all data in parallel
        with st.spinner(f"⚡ Fast Loading {len(st.session_state.watchlist)} funds..."):
            bulk_data = fetch_data_parallel(st.session_state.watchlist)

        for code in st.session_state.watchlist:
            c1, c2, c3, c4, c5, c6 = st.columns([0.5, 3, 1, 1, 1, 0.5])
            
            fund_name = schemes_dict.get(code, code)
            df_wl = bulk_data.get(code, pd.DataFrame()) # Get from bulk data
            
            # 1. Click-to-Chart Button
            if c1.button("📊", key=f"chart_{code}"):
                st.session_state.selected_scheme_code = code
                st.rerun()

            # 2. Name
            c2.write(f"**{fund_name}**")
            
            # 3. Stats
            stats = calculate_indicators(df_wl)
            if stats:
                c3.write(f"₹{stats['NAV']:.2f}")
                c3.caption(stats['Date'])
                c4.write(f"14: **{stats['RSI_D_14']}**")
                c5.write(stats['EMA_Signal'])
            else:
                c3.write("Error")
            
            # 4. Delete
            if c6.button("🗑️", key=f"del_{code}"):
                st.session_state.watchlist.remove(code)
                save_watchlist(st.session_state.watchlist)
                st.rerun()
            
            st.divider()
    else:
        st.info("Watchlist is empty.")

#streamlit run "D:\invest\mutual fund with gemini\perplexity.py"

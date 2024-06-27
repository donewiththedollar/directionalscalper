import streamlit as st
import pandas as pd
import time
import json
import plotly.express as px
import os
import tempfile

# Setting the Streamlit page configuration
st.set_page_config(layout="wide", page_title="DirectionalScalper")
st.title("DirectionalScalper Dashboard 🤖")

def write_to_json(data: dict, filename: str):
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        json.dump(data, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
    os.rename(tmp.name, filename)

def save_symbol_data(data: pd.DataFrame):
    # Convert DataFrame to a dictionary for JSON serialization
    data_dict = data.to_dict(orient='records')
    symbols_dict = {entry['symbol']: entry for entry in data_dict}
    
    # Use the atomic write function to save the data
    write_to_json(symbols_dict, "../../data/shared_data.json")

def save_open_positions_data(data: pd.DataFrame):
    # Convert DataFrame to a dictionary for JSON serialization
    data_dict = data.to_dict(orient='records')
    
    # Use the atomic write function to save the data
    write_to_json(data_dict, "../../data/open_positions_data.json")

def get_symbol_data(retries=5, delay=0.5) -> pd.DataFrame:
    json_path = "../../data/shared_data.json"  # Updated path
    for _ in range(retries):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                return pd.DataFrame(list(data.values()))
        except FileNotFoundError:
            st.error(f"File {json_path} not found.")
            return pd.DataFrame()  # Return empty DataFrame
        except json.JSONDecodeError:
            time.sleep(delay)
            continue  # Retry reading the file

    st.warning("Trouble fetching the data. Please refresh or try again later.")
    return pd.DataFrame()  # Return empty DataFrame

def get_open_positions_data() -> pd.DataFrame:
    json_path = "../../data/open_positions_data.json"  # Updated path
    with open(json_path, "r") as f:
        content = f.read()
        if not content.strip():  # Check if file is empty
            return pd.DataFrame()  # Return empty DataFrame

        try:
            open_positions = pd.DataFrame(json.loads(content))
            # Drop unnecessary columns
            open_positions = open_positions.drop(columns=["info", "id", "lastUpdateTimestamp", "percentage", "lastPrice", "contractSize", "datetime", "timestamp", "maintenanceMarginPercentage", "initialMarginPercentage", "maintenanceMargin", "marginRatio"])
            return open_positions
        except json.JSONDecodeError:
            return pd.DataFrame()  # Return empty DataFrame if there's a decode error

def get_open_symbols_count() -> int:
    json_path = "../../data/open_symbols_count.json"  # Updated path
    with open(json_path, "r") as f:
        content = f.read()
        if not content.strip():  # Check if file is empty
            return 0  # Return 0 if file is empty

        try:
            data = json.loads(content)
            return data["count"]
        except json.JSONDecodeError:
            return 0  # Return 0 if there's a decode error

# Password Protection
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

def authenticate(password):
    if password == st.secrets["password"]["password"]:
        st.session_state.authenticated = True
    else:
        st.error("Incorrect password")

if not st.session_state.authenticated:
    pwd = st.text_input("Enter Password", type="password")
    if st.button("Login"):
        authenticate(pwd)
    st.stop()

# Sidebar components to set the refresh rate and auto-refresh toggle
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
auto_refresh = st.sidebar.checkbox("Auto-Refresh", True)

# Calling the functions to get the data
symbol_data = get_symbol_data()
open_positions_data = get_open_positions_data()

# Create tabs for the dashboard
tabs = ["Overview", "Symbol Analysis", "Symbol Performance", "Open Positions", "Bot Control"]
selected_tab = st.sidebar.radio("Choose a Tab", tabs)

# Overview tab
if selected_tab == "Overview":
    st.header("Overview")

    # Display the count of open symbols
    open_symbols_count = get_open_symbols_count()
    st.metric("Open Symbols Count", f"{open_symbols_count}")
    
    total_balance = symbol_data["balance"].iloc[0]
    st.metric("Total Balance", f"${total_balance:,.2f}")
    total_long_upnl = symbol_data["long_upnl"].sum()
    st.metric("Total Long uPNL", f"${total_long_upnl:,.2f}")
    total_short_upnl = symbol_data["short_upnl"].sum()
    st.metric("Total Short uPNL", f"${total_short_upnl:,.2f}")

    st.header("Charts")
    fig_price = px.line(symbol_data, x="symbol", y="current_price", title="Price Trend over Symbols")
    st.plotly_chart(fig_price)

    fig_upnl = px.bar(symbol_data, x="symbol", y=["long_upnl", "short_upnl"], title="Long and Short uPNL for Symbols")
    st.plotly_chart(fig_upnl)

    fig_balance = px.pie(symbol_data, names="symbol", values="balance", title="Balance Distribution over Symbols")
    st.plotly_chart(fig_balance)

# Symbol Analysis tab
elif selected_tab == "Symbol Analysis":
    st.header("Symbol Analysis")
    symbol = st.selectbox("Choose a symbol", symbol_data["symbol"].unique())
    selected_data = symbol_data[symbol_data["symbol"] == symbol].iloc[0]
    for key, value in selected_data.items():
        st.write(f"{key}: {value}")

# Symbol Performance tab
elif selected_tab == "Symbol Performance":
    st.header("Symbol Performance")
    fig_volume = px.bar(symbol_data, x="symbol", y="volume", title="Volume for each Symbol")
    st.plotly_chart(fig_volume)

    fig_spread = px.bar(symbol_data, x="symbol", y="spread", title="Spread for each Symbol")
    st.plotly_chart(fig_spread)

    fig_trend = px.pie(symbol_data, names="trend", title="Trend Distribution")
    st.plotly_chart(fig_trend)

# Open Positions tab
elif selected_tab == "Open Positions":
    st.header("Open Positions")
    st.write(open_positions_data)

# Bot Control tab
elif selected_tab == "Bot Control":
    st.header("Bot Control Panel 🎮")
    
    # Enhanced button styling
    st.markdown("""
    <style>
    .control-button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stop-button {
        background-color: #f44336; /* Red */
    }
    </style>
    """, unsafe_allow_html=True)

    start_bot_button = st.markdown('<button class="control-button">Start Bot</button>', unsafe_allow_html=True)
    stop_bot_button = st.markdown('<button class="control-button stop-button">Stop Bot</button>', unsafe_allow_html=True)
    
    if start_bot_button:
        # Call function to start the bot
        st.success("Bot Started!")
    
    if stop_bot_button:
        # Call function to stop the bot
        st.warning("Bot Stopped!")
    
    strategy_param = st.slider("Set Strategy Parameter", 0, 100)
    st.write(f"You set the strategy parameter to {strategy_param}")
    # Use this strategy_param value in your bot

# Displaying the detailed symbol data table
st.header("Live Symbol Data")
st.write(symbol_data)

# Logic for auto-refreshing the dashboard at the selected interval
if auto_refresh:
    time.sleep(refresh_rate)
    st.experimental_rerun()

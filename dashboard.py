# Import necessary libraries
import streamlit as st
import pandas as pd
import time
import json
import plotly.express as px
import os
import tempfile

# Setting the Streamlit page configuration
st.set_page_config(layout="wide", page_title="DirectionalSca;per")
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
    write_to_json(symbols_dict, "shared_data.json")

def save_open_positions_data(data: pd.DataFrame):
    # Convert DataFrame to a dictionary for JSON serialization
    data_dict = data.to_dict(orient='records')
    
    # Use the atomic write function to save the data
    write_to_json(data_dict, "open_positions_data.json")

def get_symbol_data() -> pd.DataFrame:
    with open("shared_data.json", "r") as f:
        content = f.read()
        if not content.strip():  # Check if file is empty
            return pd.DataFrame()  # Return empty DataFrame

        try:
            data = json.loads(content)
            return pd.DataFrame(list(data.values()))
        except json.JSONDecodeError:
            return pd.DataFrame()  # Return empty DataFrame if there's a decode error

def get_open_positions_data() -> pd.DataFrame:
    with open("open_positions_data.json", "r") as f:
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

# Sidebar components to set the refresh rate and auto-refresh toggle
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
auto_refresh = st.sidebar.checkbox("Auto-Refresh", True)

# Calling the functions to get the data
symbol_data = get_symbol_data()
open_positions_data = get_open_positions_data()

# Create tabs for the dashboard
tabs = ["Overview", "Symbol Analysis", "Symbol Performance", "Open Positions"]
selected_tab = st.sidebar.radio("Choose a Tab", tabs)

# Overview tab
if selected_tab == "Overview":
    st.header("Overview")
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

# Displaying the detailed symbol data table
st.header("Rotator Symbol Data")
st.write(symbol_data)

# Logic for auto-refreshing the dashboard at the selected interval
if auto_refresh:
    time.sleep(refresh_rate)
    st.experimental_rerun()


# # Import necessary libraries
# import streamlit as st
# import pandas as pd
# import time
# import json
# import plotly.express as px
# import os
# import tempfile

# # Setting the Streamlit page configuration
# st.set_page_config(layout="wide", page_title="Trade Simple")
# st.title("Bot Dashboard 🤖")

# def write_to_json(data: dict, filename: str):
#     with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
#         json.dump(data, tmp)
#         tmp.flush()
#         os.fsync(tmp.fileno())
#     os.rename(tmp.name, filename)

# def save_symbol_data(data: pd.DataFrame):
#     # Convert DataFrame to a dictionary for JSON serialization
#     data_dict = data.to_dict(orient='records')
#     symbols_dict = {entry['symbol']: entry for entry in data_dict}
    
#     # Use the atomic write function to save the data
#     write_to_json(symbols_dict, "shared_data.json")

# def get_symbol_data() -> pd.DataFrame:
#     with open("shared_data.json", "r") as f:
#         content = f.read()
#         if not content.strip():  # Check if file is empty
#             return pd.DataFrame()  # Return empty DataFrame

#         try:
#             data = json.loads(content)
#             return pd.DataFrame(list(data.values()))
#         except json.JSONDecodeError:
#             return pd.DataFrame()  # Return empty DataFrame if there's a decode error


# def get_open_positions_data() -> pd.DataFrame:
#     with open("open_positions_data.json", "r") as f:
#         content = f.read()
#         if not content.strip():  # Check if file is empty
#             return pd.DataFrame()  # Return empty DataFrame

#         try:
#             open_positions = pd.DataFrame(json.loads(content))
#             # Drop unnecessary columns
#             open_positions = open_positions.drop(columns=["info", "id", "lastUpdateTimestamp", "percentage", "lastPrice", "contractSize", "datetime", "timestamp", "maintenanceMarginPercentage", "initialMarginPercentage", "maintenanceMargin", "marginRatio"])
#             return open_positions
#         except json.JSONDecodeError:
#             return pd.DataFrame()  # Return empty DataFrame if there's a decode error

# # # Function to get symbol data from the shared_data.json file
# # def get_symbol_data() -> pd.DataFrame:
# #     with open("shared_data.json", "r") as f:
# #         return pd.DataFrame(list(json.load(f).values()))

# # # Function to get open positions data from the open_positions_data.json file
# # def get_open_positions_data() -> pd.DataFrame:
# #     with open("open_positions_data.json", "r") as f:
# #         open_positions = pd.DataFrame(json.load(f))
# #         # Drop unnecessary columns
# #         open_positions = open_positions.drop(columns=["info", "id", "lastUpdateTimestamp", "percentage", "lastPrice", "contractSize", "datetime", "timestamp","maintenanceMarginPercentage", "initialMarginPercentage", "maintenanceMargin","marginRatio"])
# #         return open_positions

# # Sidebar components to set the refresh rate and auto-refresh toggle
# refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
# auto_refresh = st.sidebar.checkbox("Auto-Refresh", True)

# # Calling the functions to get the data
# symbol_data = get_symbol_data()
# open_positions_data = get_open_positions_data()

# # Create tabs for the dashboard
# tabs = ["Overview", "Symbol Analysis", "Symbol Performance", "Open Positions"]
# selected_tab = st.sidebar.radio("Choose a Tab", tabs)

# # Overview tab
# if selected_tab == "Overview":
#     st.header("Overview")
#     total_balance = symbol_data["balance"].iloc[0]
#     st.metric("Total Balance", f"${total_balance:,.2f}")
#     total_long_upnl = symbol_data["long_upnl"].sum()
#     st.metric("Total Long uPNL", f"${total_long_upnl:,.2f}")
#     total_short_upnl = symbol_data["short_upnl"].sum()
#     st.metric("Total Short uPNL", f"${total_short_upnl:,.2f}")

#     st.header("Charts")
#     fig_price = px.line(symbol_data, x="symbol", y="current_price", title="Price Trend over Symbols")
#     st.plotly_chart(fig_price)

#     fig_upnl = px.bar(symbol_data, x="symbol", y=["long_upnl", "short_upnl"], title="Long and Short uPNL for Symbols")
#     st.plotly_chart(fig_upnl)

#     fig_balance = px.pie(symbol_data, names="symbol", values="balance", title="Balance Distribution over Symbols")
#     st.plotly_chart(fig_balance)

# # Symbol Analysis tab
# elif selected_tab == "Symbol Analysis":
#     st.header("Symbol Analysis")
#     symbol = st.selectbox("Choose a symbol", symbol_data["symbol"].unique())
#     selected_data = symbol_data[symbol_data["symbol"] == symbol].iloc[0]
#     for key, value in selected_data.items():
#         st.write(f"{key}: {value}")

# # Symbol Performance tab
# elif selected_tab == "Symbol Performance":
#     st.header("Symbol Performance")
#     fig_volume = px.bar(symbol_data, x="symbol", y="volume", title="Volume for each Symbol")
#     st.plotly_chart(fig_volume)

#     fig_spread = px.bar(symbol_data, x="symbol", y="spread", title="Spread for each Symbol")
#     st.plotly_chart(fig_spread)

#     fig_trend = px.pie(symbol_data, names="trend", title="Trend Distribution")
#     st.plotly_chart(fig_trend)

# # Open Positions tab
# elif selected_tab == "Open Positions":
#     st.header("Open Positions")
#     st.write(open_positions_data)

# # Displaying the detailed symbol data table
# st.header("Rotator Symbol Data")
# st.write(symbol_data)

# # Logic for auto-refreshing the dashboard at the selected interval
# if auto_refresh:
#     time.sleep(refresh_rate)
#     st.experimental_rerun()

# import streamlit as st
# import pandas as pd
# import time
# import json
# import plotly.express as px

# st.set_page_config(layout="wide", page_title="Trade Simple")
# st.title("Bot Dashboard 🤖")

# def get_latest_data() -> pd.DataFrame:
#     # Load data from JSON file
#     with open("shared_data.json", "r") as f:
#         shared_symbols_data = json.load(f)
    
#     data = []
#     for symbol_data in shared_symbols_data.values():
#         row = [
#             symbol_data['symbol'],
#             symbol_data.get('min_qty', 0),
#             symbol_data.get('current_price', 0),
#             symbol_data.get('balance', 0),
#             symbol_data.get('available_bal', 0),
#             symbol_data.get('volume', 0),
#             symbol_data.get('spread', 0),
#             symbol_data.get('trend', ''),
#             symbol_data.get('long_pos_qty', 0),
#             symbol_data.get('short_pos_qty', 0),
#             symbol_data.get('long_upnl', 0),
#             symbol_data.get('short_upnl', 0),
#             symbol_data.get('long_cum_pnl', 0),
#             symbol_data.get('short_cum_pnl', 0),
#             symbol_data.get('long_pos_price', 0),
#             symbol_data.get('short_pos_price', 0)
#         ]
#         data.append(row)

#     columns = [
#         "Symbol", "Min. Qty", "Price", "Balance", "Available Bal.", 
#         "1m Vol", "5m Spread", "Trend", "Long Pos. Qty", "Short Pos. Qty", 
#         "Long uPNL", "Short uPNL", "Long cum. uPNL", "Short cum. uPNL", 
#         "Long Pos. Price", "Short Pos. Price"
#     ]
#     return pd.DataFrame(data, columns=columns)

# # Sidebar components
# refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
# auto_refresh = st.sidebar.checkbox("Auto-Refresh", True)
# data = get_latest_data()

# # Create tabs
# tabs = ["Overview", "Symbol Analysis", "Symbol Performance"]
# selected_tab = st.sidebar.radio("Choose a Tab", tabs)

# if selected_tab == "Overview":
#     # Overview Section
#     st.header("Overview")
#     total_balance = data["Balance"].iloc[0]
#     st.metric("Total Balance", f"${total_balance:,.2f}")
#     total_long_upnl = data["Long uPNL"].sum()
#     st.metric("Total Long uPNL", f"${total_long_upnl:,.2f}")
#     total_short_upnl = data["Short uPNL"].sum()
#     st.metric("Total Short uPNL", f"${total_short_upnl:,.2f}")

#     # Charts Section
#     st.header("Charts")
#     # Price Trend Line Chart
#     fig_price = px.line(data, x="Symbol", y="Price", title="Price Trend over Symbols")
#     st.plotly_chart(fig_price)
#     # Long and Short uPNL Bar Chart
#     fig_upnl = px.bar(data, x="Symbol", y=["Long uPNL", "Short uPNL"], title="Long and Short uPNL for Symbols")
#     st.plotly_chart(fig_upnl)
#     # Balance Distribution Pie Chart
#     fig_balance = px.pie(data, names="Symbol", values="Balance", title="Balance Distribution over Symbols")
#     st.plotly_chart(fig_balance)

# elif selected_tab == "Symbol Analysis":
#     # Symbol Analysis
#     st.header("Symbol Analysis")
#     symbol = st.selectbox("Choose a symbol", data["Symbol"].unique())
#     selected_data = data[data["Symbol"] == symbol].iloc[0]
#     for key, value in selected_data.items():
#         st.write(f"{key}: {value}")

# elif selected_tab == "Symbol Performance":
#     # Symbol Performance
#     st.header("Symbol Performance")
#     # Bar chart for volume
#     fig_volume = px.bar(data, x="Symbol", y="1m Vol", title="Volume for each Symbol")
#     st.plotly_chart(fig_volume)
#     # Bar chart for spread
#     fig_spread = px.bar(data, x="Symbol", y="5m Spread", title="Spread for each Symbol")
#     st.plotly_chart(fig_spread)
#     # Pie chart for trend distribution
#     fig_trend = px.pie(data, names="Trend", title="Trend Distribution")
#     st.plotly_chart(fig_trend)

# # Data Table Section
# st.header("Detailed Data")
# st.write(data)

# # Auto-refresh logic
# if auto_refresh:
#     time.sleep(refresh_rate)
#     st.experimental_rerun()
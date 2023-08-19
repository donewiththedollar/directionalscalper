import streamlit as st
import pandas as pd
import time
import json
import plotly.express as px

st.set_page_config(layout="wide", page_title="Trade Simple")
st.title("Bot Dashboard 🤖")

def get_latest_data() -> pd.DataFrame:
    # Load data from JSON file
    with open("shared_data.json", "r") as f:
        shared_symbols_data = json.load(f)
    
    data = []
    for symbol_data in shared_symbols_data.values():
        row = [
            symbol_data['symbol'],
            symbol_data.get('min_qty', 0),
            symbol_data.get('current_price', 0),
            symbol_data.get('balance', 0),
            symbol_data.get('available_bal', 0),
            symbol_data.get('volume', 0),
            symbol_data.get('spread', 0),
            symbol_data.get('trend', ''),
            symbol_data.get('long_pos_qty', 0),
            symbol_data.get('short_pos_qty', 0),
            symbol_data.get('long_upnl', 0),
            symbol_data.get('short_upnl', 0),
            symbol_data.get('long_cum_pnl', 0),
            symbol_data.get('short_cum_pnl', 0),
            symbol_data.get('long_pos_price', 0),
            symbol_data.get('short_pos_price', 0)
        ]
        data.append(row)

    columns = [
        "Symbol", "Min. Qty", "Price", "Balance", "Available Bal.", 
        "1m Vol", "5m Spread", "Trend", "Long Pos. Qty", "Short Pos. Qty", 
        "Long uPNL", "Short uPNL", "Long cum. uPNL", "Short cum. uPNL", 
        "Long Pos. Price", "Short Pos. Price"
    ]
    return pd.DataFrame(data, columns=columns)

# Sidebar components
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
auto_refresh = st.sidebar.checkbox("Auto-Refresh", True)

data = get_latest_data()

# Overview Section
st.header("Overview")
total_balance = data["Balance"].iloc[0]
st.metric("Total Balance", f"${total_balance:,.2f}")
total_long_upnl = data["Long uPNL"].sum()
st.metric("Total Long uPNL", f"${total_long_upnl:,.2f}")
total_short_upnl = data["Short uPNL"].sum()
st.metric("Total Short uPNL", f"${total_short_upnl:,.2f}")

# Charts Section
st.header("Charts")

# Price Trend Line Chart
fig_price = px.line(data, x="Symbol", y="Price", title="Price Trend over Symbols")
st.plotly_chart(fig_price)

# Long and Short uPNL Bar Chart
fig_upnl = px.bar(data, x="Symbol", y=["Long uPNL", "Short uPNL"], title="Long and Short uPNL for Symbols")
st.plotly_chart(fig_upnl)

# Balance Distribution Pie Chart
fig_balance = px.pie(data, names="Symbol", values="Balance", title="Balance Distribution over Symbols")
st.plotly_chart(fig_balance)

# Data Table Section
st.header("Detailed Data")
st.write(data)

# Auto-refresh logic
if auto_refresh:
    time.sleep(refresh_rate)
    st.experimental_rerun()


# import streamlit as st
# import pandas as pd
# import time
# import json


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

# # Create columns
# left_column, main_column, right_column = st.columns([1, 6, 1])  # You can adjust the ratios

# # Display the data table in the main column
# with main_column:
#     st.write(get_latest_data())

# # Add auto-refresh functionality
# refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 5, 60, 10)
# auto_refresh = st.sidebar.checkbox("Auto-Refresh", True)

# if auto_refresh:
#     time.sleep(refresh_rate)
#     st.experimental_rerun()
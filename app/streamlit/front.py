import streamlit as st
import yfinance as yf
import datetime
import plotly.graph_objects as go
import requests
import json 
import pandas as pd

API_URL = "http://fastapi:8000/LSTM_Predict"

min_date = datetime.datetime(2020, 1, 1)
max_date = datetime.datetime.now()

stock_tick = st.selectbox("Enter Stock Ticker", ("MSFT"))

start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

if start_date > end_date:
    st.error("Start date must be before end date.")
else:
    st.success("Start date: {} End date: {}".format(start_date, end_date))

stock_data = yf.download(stock_tick, start=start_date, end=end_date)
stock_data.reset_index(inplace=True)
stock_data["Date"] = pd.to_datetime(stock_data['Date'])

print(stock_data)

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data['Date'], y=stock_data["Close"], name="Close"))
fig.update_layout(title=f"{stock_tick} Stock Price", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig)

if st.button("Predict"):
    payload = {"ticker": stock_tick, "start_date": start_date.strftime("%Y-%m-%d"), "end_date": end_date.strftime("%Y-%m-%d")}

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()

        prediction = json.loads(response.text)
        predicted_price = prediction["predicted_price"]
        dates = prediction["dates"]
        actual_prices = stock_data
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=stock_data.index, y=actual_prices, mode="lines", name="Actual"))
        # fig.add_trace(go.Scatter(x=[stock_data.index[-1], stock_data.index[-1] + datetime.timedelta(days=1)], y=[actual_prices[-1], predicted_price], mode="lines", name="Predicted"))
        fig.update_layout(title=f"{stock_tick} Stock Price")
        st.plotly_chart(fig)

    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")

        
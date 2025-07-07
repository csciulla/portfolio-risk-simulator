from src.portfolio import Portfolio
import streamlit as st
import pandas as pd

st.title("Stress Testing Dashboard")

#User Inputs
col1, col2 = st.columns(2)
with col1:
    #Input tickers
    tickers_input = st.text_input("Enter tickers (comma seperated):", value="AAPL, MSFT, GOOG")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
with col2:
    #Use period
    period_choices = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    period = st.selectbox("Select period:", period_choices)
    start_date = end_date = None

    #Use custom date
    custom_dates = st.checkbox("Use custom date range instead")
    if custom_dates:
        start_date=st.date_input("Start Date")
        end_date=st.date_input("End date")

weightsChoice = st.radio("Weight Allocation", ["Equal Weighted", "Optimized"])
if weightsChoice == "Equal Weighted":
    choice = 'eq'
    lbound, ubound = 0.0, 1.0
elif weightsChoice == "Optimized":
    choice = 'opt'
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        lbound = st.number_input("Enter lower bound:", min_value=0.0, max_value=0.5, value=0.0)
    with bcol2:
        ubound = st.number_input("Enter upper bound:", min_value=0.0, max_value=1.0, value=0.5)

#Download button
if st.button("Download Data"):
    port = Portfolio(tickers, lbound, ubound)
    df, df_error = port.get_data(period=period, start_date=start_date, end_date=end_date)
    weights, w_error = port.get_weights(type_weight=choice)
    if df_error:
        st.error(f"Data download failed: {df_error}")
    elif w_error:
        st.error(f"Weight calculated failed: {w_error}")
    else:
        st.write("Portfolio data", df)
        st.write("Portfolio Weights", weights)

from src.portfolio import Portfolio
import streamlit as st
import pandas as pd

st.set_page_config(page_title = "Portfolio Risk Simulator", layout="wide")

with st.sidebar:
    st.title("Portfolio Risk Simulator")
    page = st.radio('' ,["Configuration","Metrics & Visualization"])

if page == "Configuration":
    st.subheader("Portfolio Configuration")
    #Enter tickers
    tickers_input = st.text_input("**Enter Portfolio Tickers:**", 
                                placeholder='AAPL, MSFT, GOOG, TSLA', 
                                help="Enter stock symbols seperated by commas")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

    #Enter timeframe
    time_select = st.radio("**Data Retrieval Method:**", ['Quick Select', 'Custom Date Range'], horizontal=True)
    if time_select == 'Quick Select':
        start_date = end_date = None
        month_choices = [f'{i}mo' for i in range(1,12)]
        year_choices = [f'{j}y' for j in range(1,51)]
        full_choices = month_choices + year_choices + ['ytd', 'max']

        period = st.select_slider("Choose a period:", full_choices)

    elif time_select == 'Custom Date Range':
        period = None
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            start_date = st.date_input("Enter start date:")
        with dcol2:
            end_date = st.date_input("Enter end date:")

    #Enter weights
    weights_input = st.radio("**Weight Allocation:**",
                            ['Equal Weighted', 'Optimized', 'Custom'],
                            horizontal=True, help="Optimized maximizes the weights according to the Sharpe Ratio")
    lbound, ubound = 0.0, 1.0

    if weights_input == 'Equal Weighted':
        type_weight = 'eq'
    elif weights_input == 'Optimized':
        type_weight = 'opt'
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            lbound = st.number_input("Enter lower bound:", min_value=0.0, max_value=0.5, value=0.0)
        with bcol2:
            ubound = st.number_input("Enter upper bound:", min_value=0.0, max_value=1.0, value=0.5)
    else:
        placeholder_weights = ', '.join('0'*len(tickers))
        custom_weights = st.text_input("Enter weights (comma seperated):", placeholder_weights)
        weights = [pd.to_numeric(weight) for weight in custom_weights.split(",")]

    #Download button
    if st.button("Download Data"):
        port = Portfolio(tickers, lbound, ubound)
        df, df_error = port.get_data(period=period, start_date=start_date, end_date=end_date)
        weights, w_error = port.get_weights(type_weight=type_weight)
        if df_error:
            st.error(f"Data download failed: {df_error}")
        elif w_error:
            st.error(f"Weight calculated failed: {w_error}")
        else:
            st.success("Portfolio data successfully downloaded!")

    st.markdown("---")

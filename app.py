from src.portfolio import Portfolio
from src.simulation import monte_carlo, historical, FactorStress
import streamlit as st
import pandas as pd

#Page config and session state initalization
st.set_page_config(page_title = "Portfolio Risk Simulator", layout="wide")
if 'df' not in st.session_state:
    st.session_state.df = None
if 'port' not in st.session_state:
    st.session_state.port = None
if 'weights' not in st.session_state:
    st.session_state.weights = None

def render_sidebar():
    """
    Creates sidebar for UI.
    """
    with st.sidebar:
        st.title("Portfolio Risk Simulator")
        page = st.radio('' ,["Portfolio Configuration","Simulation Configuration","Metrics & Visualization"])
        return page

def portfolio_config():
    """ 
    UI for Portfolio Configuration page.
    """
    pcol1, pcol2 = st.columns(2)

    #Input Parameters
    with pcol1:
        st.subheader("Parameters")
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
            with st.spinner("Downloading portfolio data..."):
                try:
                    port = Portfolio(tickers, lbound, ubound)
                    df, df_error = port.get_data(period=period, start_date=start_date, end_date=end_date)
                    weights, w_error = port.get_weights(type_weight=type_weight)
                    _, norm_error = port.normalize_portfolio_data()
                    _, c_error = port.get_portfolio_colors()

                    if df_error:
                        st.error(f"Error in downloading portfolio data: {df_error}")
                    elif w_error:
                        st.error(f"Error in calculating weights: {w_error}")
                    elif norm_error:
                        st.error(f"Error in normalizing portfolio data: {norm_error}")
                    elif c_error:
                        st.error(f"Error in constructing asset colors: {c_error}")
                    else:
                        st.session_state.df = df
                        st.session_state.port = port
                        st.session_state.weights = weights
                        #st.success("Portfolio data successfully downloaded!")
                                        
                except Exception as e:
                    st.error(f"Portfolio initalization failed: {e}")    
  
    with pcol2:
        #Weight Pie Chart
        if st.session_state.df is None:
            st.info("📊 Portfolio pie chart will appear here once data is downloaded.")
        elif st.session_state.port is not None:
            pie, pie_error = port.plot_pie()
            if pie_error:
                st.error(f"Error in plotting weight allocation: {pie_error}")
            else:
                st.plotly_chart(pie, use_container_width=True)

    #Portofolio Line Chart
    if st.session_state.df is None:
         st.info("📊 Portfolio line chart will appear here once data is downloaded.")
    elif st.session_state.port is not None:
        line, line_error = port.plot_line()
        if line_error:
            st.error(f"Error in plotting portfolio data: {line_error}")
        else:
            st.plotly_chart(line, use_container_width=True)

        

def simulation_config():
    """ 
    UI for Simulation Configuration page.
    """
    #Simulation Configuration
    st.subheader("Simulation Configuration")
    sim_method = st.radio("**Simulation Method:**", ['Monte Carlo', 'Historical Replay'], horizontal=True)
    if sim_method == 'Monte Carlo':
                
        #Factor setup
        factors = None
        classifyq = st.checkbox("**Classify your portfolio?**", )
        if classifyq:
            factors_expanded_select = st.multiselect("Select Factors:", ['3-factor Fama-French', 
                                                                        '5-factor Fama-French',
                                                                        'Market Premium',
                                                                        'Size Premium',
                                                                        'Value Premium',
                                                                        'Profitability Factor',
                                                                        'Investment Factor',
                                                                        'Momentum Effect'])
                        
            factor_convert = {'3-factor Fama-French': 'FF3',
                            '5-factor Fama-French': 'FF5',
                            'Market Premium': 'Mkt-RF',
                            'Size Premium': 'SMB',
                            'Value Premium': 'HML',
                            'Profitability Factor': 'RMW',
                            'Investment Factor': 'CMA',
                            'Momentum Effect': 'Mom'}
                        
            factors = [factor_convert[factor] for factor in factors_expanded_select] 

def main():
    page = render_sidebar()

    if page == 'Portfolio Configuration':
        portfolio_config()
    elif page == 'Simulation Configuration':
        simulation_config()
    elif page == 'Metrics & Visualization':
        pass

if __name__ == '__main__':
    main()

            

from app.common import * 
from portfolio import Portfolio
from app.ui.utils_ui import render_info_box
from app.cached_functions import cached_get_data, cached_get_weights
import pandas as pd
import streamlit as st
import time
from datetime import date


#========== PORFOLIO UI FUNCTIONS =========

def render_basic_config(portfolios:dict, name:str):
    """
    Renders portfolio ticker and value input components.
    Displays locked state with previously selected values after portfolio download.

    Parameters:
    - portfolios: Dictionary containing all characteristics for each portfolio that the user has defined
    - name: String of the current portfolio name
    """
    if portfolios[name]['configs']['tickers'] is None and portfolios[name]['configs']['portfolio_value'] is None:
        tickers_input = st.text_input("**Enter Portfolio Tickers:**", 
                                      placeholder='AAPL, MSFT, GOOG, TSLA', 
                                      help="Enter stock symbols seperated by commas")
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

        portfolio_value = st.number_input("**Enter Portfolio Value ($):**",
                                          value=100000, 
                                          min_value=1000,
                                          max_value=10000000, 
                                          step=5000,
                                          format='%d',
                                          help='Enter your inital investment amount in USD')
    
    #Display locked state if tickers already chosen 
    else:
        tickers = st.text_input("**Enter Portfolio Tickers:**",
                                value=", ".join(portfolios[name]['configs']['tickers']),
                                disabled=True,
                                help="Previously configured tickers")
        
        portfolio_value = st.number_input("**Enter Portfolio Value ($):**",
                                          value=portfolios[name]['configs']['portfolio_value'],
                                          disabled=True,
                                          help="Previously configured portfolio value")

    return tickers, portfolio_value


def render_timeframe_input(portfolios:dict, name:str):
    """
    Renders data retrieval method selection and timeframe configuration.
    Displays locked state with previously selected values after portfolio download.

    Parameters:
    - portfolios: Dictionary containing all characteristics for each portfolio that the user has defined
    - name: String of the current portfolio name
    """
    month_choices = [f'{i}mo' for i in range(1,12)]
    year_choices = [f'{j}y' for j in range(1,51)]
    full_choices = month_choices + year_choices + ['ytd', 'max']
    
    if portfolios[name]['configs']['timeframe'] is None:
        time_select = st.radio("**Data Retrieval Method:**", ['Quick Select', 'Custom Date Range'], horizontal=True)
        
        if time_select == 'Quick Select':
            start_date = end_date = None
            period = st.select_slider("Choose a period:", full_choices, value='1y')
            if period in ['1mo', '2mo', '3mo']:
                st.warning("Factor classification will be unable due an insufficient amount of data.", icon="⚠️")

        elif time_select == 'Custom Date Range':
            period = None
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                start_date = st.date_input("Enter start date:", value=date(2020, 1, 1), min_value=date(1990, 1, 1))
            with dcol2:
                end_date = st.date_input("Enter end date:", value=date.today(), min_value=date(1990, 1, 1), max_value=date.today())
            
            if (end_date - start_date).days < 70:
                st.warning("Factor classification will be unable due an insufficient amount of data.", icon="⚠️")

    #Display locked state if timeframe already chosen
    else:
        if isinstance(portfolios[name]['configs']['timeframe'], str):
            start_date = end_date = None
            st.radio("**Data Retrieval Method:**", 
                     options=['Quick Select', 'Custom Date Range'], 
                     index=0,
                     horizontal=True,
                     disabled=True, 
                     help="Previously chosen retrieval method")
                    
            period = st.select_slider("Choose a period:", 
                                      full_choices, 
                                      value=portfolios[name]['configs']['timeframe'],
                                      disabled=True)

        elif isinstance(portfolios[name]['configs']['timeframe'], tuple):
            period = None
            st.radio("**Data Retrieval Method:**", 
                     options=['Quick Select', 'Custom Date Range'],
                     index=1,
                     horizontal=True,
                     disabled=True, 
                     help="Previously chosen retrieval method")
                    
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                start_date = st.date_input("Enter start date:",
                                           value=portfolios[name]['configs']['timeframe'][0],
                                           disabled=True,
                                           help="Previously chosen start date")
            with dcol2:
                end_date = st.date_input("Enter end date:",
                                         value=portfolios[name]['configs']['timeframe'][1],
                                         disabled=True,
                                         help="Previosuly chosen end date")
                    
    return period, start_date, end_date


def render_weight_input(portfolios:dict, name:str, tickers:list):
    """
    Renders weight allocation components.
    Displays locked state with previously selected values after portfolio download.
    """
    
    #Enter weights 
    if portfolios[name]['configs']['weight_allocation']['method'] is None: 
        weights_input = st.radio("**Weight Allocation:**",
                                 ['Equal Weighted', 'Optimized', 'Custom'],
                                 horizontal=True, 
                                 help="Optimized maximizes the weights according to the Sharpe Ratio")
        
        custom_weights_list = None
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
        elif weights_input == 'Custom':
            type_weight = 'custom'
            placeholder_weights = ', '.join([f'{1/len(tickers)}']*len(tickers))
            custom_weights = st.text_input("Enter weights:", placeholder=placeholder_weights, help="Enter weights comma seperated")
            custom_weights_list = [pd.to_numeric(weight.strip()) for weight in custom_weights.split(",")]
        
    #Display locked state if weights already chosen
    else:
        custom_weights_list = None
        lbound, ubound = 0.0, 1.0
        if portfolios[name]['configs']['weight_allocation']['method'] == 'eq':
            type_weight = 'eq'
            st.radio("**Weight Allocation:**",
                     ['Equal Weighted', 'Optimized', 'Custom'],
                     index=0,
                     disabled=True,
                     horizontal=True,
                     help="Previously chosen weight allocation method")
                    
        elif portfolios[name]['configs']['weight_allocation']['method'] == 'opt':
            type_weight = 'opt'
            st.radio("**Weight Allocation:**",
                     ['Equal Weighted', 'Optimized', 'Custom'],
                     index=1,
                     disabled=True,
                     horizontal=True,
                     help="Previously chosen weight allocation method")
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                lbound = st.number_input("Enter lower bound:", 
                                         value=portfolios[name]['configs']['weight_allocation']['lbound'],
                                         disabled=True,
                                         help="Previously chosen lower bound")
            with bcol2:
                ubound = st.number_input("Enter upper bound:", 
                                         value=portfolios[name]['configs']['weight_allocation']['ubound'],
                                         disabled=True,
                                         help="Previously chosen upper bound")
                    
        elif portfolios[name]['configs']['weight_allocation']['method'] == 'custom':
            type_weight = 'custom'
            st.radio("**Weight Allocation:**",
                     ['Equal Weighted', 'Optimized', 'Custom'],
                     index=2,
                     disabled=True,
                     horizontal=True,
                     help="Previously chosen weight allocation method")
            custom_weights = ', '.join([str(weight) for weight in portfolios[name]['configs']['weights']])
            custom_weights_list = st.text_input("**Enter Weights:**", 
                                                value=custom_weights,
                                                disabled=True,
                                                help="Previously chosen custom weights")
            
    return type_weight, lbound, ubound, custom_weights_list


def render_pie(portfolios:dict, name:str):
    """
    Renders pie chart.
    Returns an info box before portfolio download.
    """
    with st.container(border=True, height=560):
        st.markdown("#### Weight Allocation")
        if portfolios[name]['configs']['pie'] is None:
            render_info_box("Portfolio pie chart will appear here once data is downloaded.",
                            icon='📊', height=360, margin_top="30px")
        else:
            st.plotly_chart(portfolios[name]['configs']['pie'], width='stretch')

    
def render_line(portfolios:dict, name:str):
    """ 
    Renders line chart.
    Returns an info box before portfolio download.
    """
    if portfolios[name]['configs']['line'] is None:    
        with st.container(border=True, height=250):
            st.markdown("#### Normalized Portfolio Performance")
            render_info_box("Portfolio line chart will appear here once data is downloaded.",
                        icon='📈', height=120, margin_top="0px")
    else:
        with st.container(border=True):
            st.markdown("#### Normalized Portfolio Performance")
            st.plotly_chart(portfolios[name]['configs']['line'], width='stretch')
    

def portfolio_config():
    """ 
    UI for Portfolio Configuration page.
    """
    st.header("Portfolio Configuration", divider=True)
    portfolios = st.session_state.portfolios
    name = st.session_state.current_portfolio

    pcol1, pcol2 = st.columns(2)

    #Input Parameters
    with pcol1:
        with st.container(border=True, height=560):
            st.markdown("#### Input Parameters")

            tickers, portfolio_value = render_basic_config(portfolios, name)
            period, start_date, end_date = render_timeframe_input(portfolios, name)
            type_weight, lbound, ubound, custom_weights_list = render_weight_input(portfolios, name, tickers)

            #Download button
            if st.button("Download Portfolio Data", width='stretch'):
                with st.spinner("Downloading portfolio data..."):
                    try:
                        #Use cached data download
                        df, df_error, df_warning = cached_get_data(tickers, lbound, ubound, period=period, start_date=start_date, end_date=end_date)
                        if df_warning:
                            st.warning(df_warning, icon="⚠️")

                        #Get weights
                        if type_weight == 'custom' and custom_weights_list:
                            weights, w_error = cached_get_weights(tickers, lbound, ubound, period, start_date, end_date, type_weight, custom_weights_list)
                        else:
                            weights, w_error = cached_get_weights(tickers, lbound, ubound, period, start_date, end_date, type_weight)

                        #Initalize port for plotting functions
                        port = Portfolio(tickers, lbound, ubound)
                        port.portfolio_df = df
                        port.weights = weights

                        #Process portfolio data
                        _, norm_error = port.normalize_portfolio_data()
                        _, c_error = port.get_portfolio_colors()
    
                        #Create plots
                        pie_chart, pie_error = port.plot_pie()
                        line_chart, line_error = port.plot_line()

                        #Error checks
                        if df_error:
                            st.error(f"Error in downloading portfolio data: {df_error}", icon='❌')
                        elif w_error:
                            st.error(f"Error in calculating weights: {w_error}", icon='❌')
                        elif norm_error:
                            st.error(f"Error in normalizing portfolio data: {norm_error}", icon='❌')
                        elif c_error:
                            st.error(f"Error in constructing asset colors: {c_error}", icon='❌')
                        elif pie_error:
                            st.error(f"Error in building weight pie plot: {pie_error}", icon='❌')
                        elif line_error:
                            st.error(f"Error in building portfolio data line chart: {line_error}", icon='❌')
                        else:
                            # Store in session state
                            portfolios =  st.session_state.portfolios
                            name = st.session_state.current_portfolio
                            portfolios[name]['config_complete'] = True
                            portfolios[name]['configs']['tickers'] = tickers
                            portfolios[name]['configs']['portfolio_value'] = portfolio_value

                            if period and start_date is None and end_date is None:
                                portfolios[name]['configs']['timeframe'] = period
                            else:
                                portfolios[name]['configs']['timeframe'] = start_date, end_date

                            portfolios[name]['configs']['weight_allocation']['method'] = type_weight
                            portfolios[name]['configs']['weight_allocation']['lbound'] = lbound
                            portfolios[name]['configs']['weight_allocation']['ubound'] = ubound
                            portfolios[name]['configs']['df']['data'] = df
                            portfolios[name]['configs']['df']['warning'] = df_warning
                            portfolios[name]['configs']['weights'] = weights
                            portfolios[name]['configs']['pie'] = pie_chart
                            portfolios[name]['configs']['line'] = line_chart
                            
                            st.success("Portfolio data downloaded successfully!", icon="✅")
                            time.sleep(1)
                            st.rerun()

                    except Exception as e:
                        st.error(f"Portfolio initalization failed: {str(e)}", icon='❌')

    with pcol2:
        render_pie(portfolios, name)
    
    render_line(portfolios, name)
from src.portfolio import Portfolio
from src.simulation import monte_carlo, historical, FactorStress
import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd
from datetime import datetime
import time

#Page config and session state initalization
st.set_page_config(page_title = "Portfolio Risk Simulator", layout="wide")

if 'portfolios' not in st.session_state:
    st.session_state.portfolios = {}
if 'portfolio_counter' not in st.session_state:
    st.session_state.portfolio_counter = 1
if 'current_portfolio' not in st.session_state:
    st.session_state.current_portfolio = None

def home():
    """
    Creates the home page for portfolio initalization and user introduction.
    """
    st.title("Portfolio Risk Simulator")

    with st.container(border=True):
        port_ct = st.session_state.portfolio_counter
        placeholder_name = f"Portfolio {port_ct}"
        name = st.text_input("**Input a Portfolio Name:**", placeholder=placeholder_name, max_chars=30)
        init_port = st.button("Create Portfolio", use_container_width=True)
        if init_port:

            try:
                portfolios = st.session_state.portfolios
                if not name:
                    name = placeholder_name
                elif name in portfolios.keys():
                    raise ValueError("That portfolio name has already been taken. Try again.")
            
            except Exception as e:
                st.error(str(e), icon="❌")
                return
            
            created_date = datetime.now()
            
            portfolios[name] = {
                'created_date': created_date,
                'config_complete': False,
                'configs': {
                    'tickers': None,
                    'timeframe': None,
                    'weight_allocation': {'method': None,
                                          'lbound': None,
                                          'ubound': None},
                    'df': None,
                    'weights': None,
                    'pie': None,
                    'line': None,
                },
                'sim_complete': False,
                'metrics_complete': False,
                'scenarios': {}
                }

            st.session_state.portfolio_counter += 1
            st.session_state.current_portfolio = name
            st.success(f"Portfolio '{name}' created succesfully!", icon="✅")
            time.sleep(1)
            st.rerun()


def render_sidebar():
    """
    Creates sidebar for UI.
    """
    with st.sidebar:
        #Home Button
        home_button = st.button("🏠 Home", key='home_btn', use_container_width=True,
                                type='primary' if st.session_state.current_page == 'home' else 'secondary')
        if home_button:
            st.session_state.current_page = 'home'
            st.rerun()
        
        portfolios = st.session_state.portfolios
        if portfolios.keys() is not None:

                for name in portfolios.keys():
                    st.subheader(f'📁 {name}')

                    config_key = f"config_{name}"
                    sim_key = f"sim_{name}"
                    metrics_key = f"metrics_{name}"

                    #Configuration Button
                    config_button = st.button("⚙️ Configuration", key=config_key, use_container_width=True,
                                              type='primary' if st.session_state.current_page == config_key else 'secondary')
                    if config_button:
                        st.session_state.current_page = config_key
                        st.session_state.current_portfolio = name 
                        st.rerun()

                    #Simulation Button
                    if portfolios[name]['config_complete']:
                        sim_button = st.button("🎛️ Simulation", key=sim_key, use_container_width=True,
                                                type='primary' if st.session_state.current_page == sim_key else 'secondary')
                        if sim_button:
                            st.session_state.current_page = sim_key
                            st.session_state.current_portfolio = name
                            st.rerun()

                    #Metrics Button
                    if portfolios[name]['sim_complete']:
                        metrics_button = st.button("📊 Metrics", key=metrics_key, use_container_width=True,
                                                type='primary' if st.session_state.current_page == metrics_key else 'secondary')
                        if metrics_button:
                            st.session_state.current_page = metrics_key
                            st.session_state.current_portfolio = name
                            st.rerun()


def render_info_box(message:str, height:int = 300, margin_top:str = "0px", max_width:str = "900px",):
    """
    Draw a centered info-like box, vertically centered inside a flex container of given height.

    Parameters: 
    - height: px height of the placeholder container
    - margin_top: CSS margin-top value for pushing the whole placeholder down (e.g. "20px")
    - max_width: max width of the info box (keeps it from being full width)
    """
    bg = "#194e6b"   
    txt = "#ffffff"
    border = "#2a6b8d"

    message = message.replace(
        "📊", '<span style="font-size: 28px;">📊</span>'
        ).replace(
        "📈", '<span style="font-size: 28px;">📈</span>'
        )
    
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center; align-items:center; height:{height}px; margin-top:{margin_top};">
          <div style="
              background-color: {bg};
              color: {txt};
              padding: 16px 22px;
              border-radius: 10px;
              border: 1px solid {border};
              font-size: 16px;
              max-width: {max_width};
              text-align: center;
              box-shadow: rgba(0,0,0,0.08) 0px 2px 6px;
          ">
            {message}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_ticker_input(portfolios:dict, name:str):
    """
    Renders portfolio ticker input.
    Displays locked state with previously selected values after portfolio download.

    Parameters:
    - portfolios: Dictionary containing all characteristics for each portfolio that the user has defined
    - name: String of the current portfolio name
    """
    if portfolios[name]['configs']['tickers'] is None:
        tickers_input = st.text_input("**Enter Portfolio Tickers:**", 
                                      placeholder='AAPL, MSFT, GOOG, TSLA', 
                                      help="Enter stock symbols seperated by commas")
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    
    #Display locked state if tickers already chosen 
    else:
        tickers = st.text_input("**Enter Portfolio Tickers:**",
                                value=", ".join(portfolios[name]['configs']['tickers']),
                                disabled=True,
                                help="Previously configured tickers")
        
    return tickers


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
            period = st.select_slider("Choose a period:", full_choices)

        elif time_select == 'Custom Date Range':
            period = None
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                start_date = st.date_input("Enter start date:")
            with dcol2:
                end_date = st.date_input("Enter end date:")

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
    with st.container(border=True):
        st.markdown("#### Weight Allocation")
        if portfolios[name]['configs']['pie'] is None:
            render_info_box("📊 Portfolio pie chart will appear here once data is downloaded.",
                            height=360, margin_top="0px")
        else:
            st.plotly_chart(portfolios[name]['configs']['pie'], use_container_width=True)

    
def render_line(portfolios:dict, name:str):
    """ 
    Renders line chart.
    Returns an info box before portfolio download.
    """
    with st.container(border=True):
        st.markdown("#### Normalized Portfolio Performance")
        if portfolios[name]['configs']['line'] is None:
            render_info_box("📈 Portfolio line chart will appear here once data is downloaded.",
                        height=120, margin_top="18px")
        else:
            st.plotly_chart(portfolios[name]['configs']['line'], use_container_width=True)
    

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
        with st.container(border=True):
            st.markdown("#### Input Parameters")

            tickers = render_ticker_input(portfolios, name)
            period, start_date, end_date = render_timeframe_input(portfolios, name)
            type_weight, lbound, ubound, custom_weights_list = render_weight_input(portfolios, name, tickers)

            #Download button
            if st.button("Download Portfolio Data", use_container_width=True):
                with st.spinner("Downloading portfolio data..."):
                    try:
                        #Create portfolio object
                        port = Portfolio(tickers, lbound, ubound)
                        df, df_error = port.get_data(period=period, start_date=start_date, end_date=end_date)

                        # Get weights
                        if type_weight == 'custom' and custom_weights_list:
                            weights, w_error = port.get_weights(type_weight=type_weight, custom_weights=custom_weights_list)
                        else:
                            weights, w_error = port.get_weights(type_weight=type_weight)

                        # Process portfolio data
                        _, norm_error = port.normalize_portfolio_data()
                        _, c_error = port.get_portfolio_colors()
    
                        # Create plots
                        pie_chart, pie_error = port.plot_pie()
                        line_chart, line_error = port.plot_line()

                        #Error checks
                        if df_error:
                            st.error(f"Error in downloading portfolio data: {df_error}")
                        elif w_error:
                            st.error(f"Error in calculating weights: {w_error}")
                        elif norm_error:
                            st.error(f"Error in normalizing portfolio data: {norm_error}")
                        elif c_error:
                            st.error(f"Error in constructing asset colors: {c_error}")
                        elif pie_error:
                            st.error(f"Error in building weight pie plot: {pie_error}")
                        elif line_error:
                            st.error(f"Error in building portfolio data line chart: {line_error}")
                        else:
                            # Store in session state
                            portfolios =  st.session_state.portfolios
                            name = st.session_state.current_portfolio
                            portfolios[name]['config_complete'] = True
                            portfolios[name]['configs']['tickers'] = tickers

                            if period and start_date is None and end_date is None:
                                portfolios[name]['configs']['timeframe'] = period
                            else:
                                portfolios[name]['configs']['timeframe'] = start_date, end_date

                            portfolios[name]['configs']['weight_allocation']['method'] = type_weight
                            portfolios[name]['configs']['weight_allocation']['lbound'] = lbound
                            portfolios[name]['configs']['weight_allocation']['ubound'] = ubound
                            portfolios[name]['configs']['df'] = df
                            portfolios[name]['configs']['weights'] = weights
                            portfolios[name]['configs']['pie'] = pie_chart
                            portfolios[name]['configs']['line'] = line_chart
                            
                            st.success("Portfolio data downloaded successfully!", icon="✅")
                            time.sleep(1)
                            st.rerun()

                    except Exception as e:
                        st.error(f"Portfolio initalization failed: {str(e)}")

    with pcol2:
        render_pie(portfolios, name)
    
    render_line(portfolios, name)
        

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
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'

    render_sidebar()

    current_page = st.session_state.current_page
    if current_page == 'home':
        home()
    elif current_page.startswith('config_'):
        portfolio_config()
    elif current_page.startswith('sim_'):
        simulation_config()
    elif current_page.startswith('metrics_'):
        pass

if __name__ == '__main__':
    main()

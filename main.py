from src.portfolio import Portfolio
from src.simulation import monte_carlo, historical, FactorStress
import streamlit as st
import streamlit_shadcn_ui as ui
import pandas as pd

#Page config and session state initalization
st.set_page_config(page_title = "Portfolio Risk Simulator", layout="wide")
if 'df' not in st.session_state:
    st.session_state.df = None
if 'port' not in st.session_state:
    st.session_state.port = None
if 'weights' not in st.session_state:
    st.session_state.weights = None
if 'pie' not in st.session_state:
    st.session_state.pie = None
if 'line' not in st.session_state:
    st.session_state.line = None


def render_sidebar():
    """
    Creates sidebar for UI.
    """
    with st.sidebar:
        st.title("Portfolio Risk Simulator")
        page = st.radio('' ,["Portfolio Configuration","Simulation Configuration","Metrics & Visualization"])
        return page


def render_info_box(message: str, height: int = 300, margin_top: str = "0px", max_width: str = "900px",):
    """
    Draw a centered info-like box (theme-aware fallback), vertically centered inside a flex container of given height.

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


def portfolio_config():
    """ 
    UI for Portfolio Configuration page.
    """
    st.header("Portfolio Configuration", divider=True)

    pcol1, pcol2 = st.columns(2)

    #Input Parameters
    with pcol1:
        with st.container(border=True):
            st.markdown("#### Input Parameters")
        
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
            elif weights_input == 'Custom':
                type_weight = 'custom'
                placeholder_weights = ', '.join([f'{1/len(tickers)}']*len(tickers))
                custom_weights = st.text_input("Enter weights:", placeholder_weights, help="Enter weights comma seperated")
                custom_weights_list = [float(weight.strip()) for weight in custom_weights.split(",")]

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
                            st.session_state.df = df
                            st.session_state.port = port
                            st.session_state.weights = weights
                            st.session_state.pie = pie_chart
                            st.session_state.line = line_chart
                            
                            st.success("Portfolio data downloaded successfully!", icon="✅")

                    except Exception as e:
                        st.error(f"Portfolio initalization failed: {str(e)}")

    with pcol2:
        #Weight Pie Chart
        with st.container(border=True):
            st.markdown("#### Weight Allocation")
            if st.session_state.pie is None:
                render_info_box("📊 Portfolio pie chart will appear here once data is downloaded.",
                            height=360, margin_top="0px")
            else:
                st.plotly_chart(st.session_state.pie, use_container_width=True)

    #Portofolio Line Chart
    with st.container(border=True):
        st.markdown("#### Normalized Portfolio Performance")
        if st.session_state.line is None:
            render_info_box("📈 Portfolio line chart will appear here once data is downloaded.",
                        height=120, margin_top="18px")
        else:
            st.plotly_chart(st.session_state.line, use_container_width=True)


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

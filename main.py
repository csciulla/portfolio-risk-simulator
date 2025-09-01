import streamlit as st
from datetime import datetime
import time

#Add src folder to Python path at app startup
import sys
import os
src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
sys.path.insert(0, src_path)

from app.ui.portfolio_ui import portfolio_config
from app.ui.simulation_ui import simulation_config
from app.ui.metrics_ui import metrics

#Page config and session state initalization
st.set_page_config(page_title = "Portfolio Risk Simulator", layout="wide")

def initalize_session_state():
    if 'portfolios' not in st.session_state:
        st.session_state.portfolios = {}
    if 'portfolio_counter' not in st.session_state:
        st.session_state.portfolio_counter = 1
    if 'current_portfolio' not in st.session_state:
        st.session_state.current_portfolio = None
    if 'current_scenario' not in st.session_state:
        st.session_state.current_scenario = None
    if 'temp_sim_method' not in st.session_state:
        st.session_state.current_sim_method = None
    if 'sim_analyzer' not in st.session_state:
        st.session_state.sim_analyzer = None
    if 'current_path_type' not in st.session_state:
        st.session_state.current_path_type = None

initalize_session_state()


def home():
    """
    Creates the home page for portfolio initalization and user introduction.
    """
    st.title("Portfolio Risk Simulator")

    # Overview section
    st.markdown("""
    #### 🎯 What This Tool Does
    
    A comprehensive risk assessment platform that helps investors understand how their portfolios perform under different market conditions. 
    Built with professional-grade methodologies including Monte Carlo simulations, Hidden Markov Models, and Fama-French factor analysis.
    """)

    #Create Portfolio button
    with st.container(border=True):
        st.markdown("### 🚀 Create Portfolio")
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
                    'portfolio_value': None,
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
                'scenario_ctr': 0,
                'scenarios': {},
                'sim_analyzer': {'init': False,
                                    'object': None},
                'metrics': None
                }

            st.session_state.portfolio_counter += 1
            st.session_state.current_portfolio = name
            st.success(f"Portfolio '{name}' created succesfully!", icon="✅")
            time.sleep(1)
            st.rerun()
    
    #Example outcomes
    st.markdown("#### 💡 Example Analysis Scenarios")
    st.markdown("- Simulate how your portfolio reacts in a extreme market environment with high volatility regimes and stress multipliers")
    st.markdown("- See how a tech portfolio reacts during a replay of COVID-19 based on data from the past 10 years")
    st.markdown("- Stress your portfolio by factors such as size and value in medium volatility regime")

    #Connect 
    st.markdown("---")
    st.markdown("##### 👤 Connect")
    st.markdown("""
    <div style="display: flex; gap: 10px; align-items: center;">
        <a href="https://linkedin.com/in/christian-sciulla" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
        </a>
        <a href="https://github.com/csciulla" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
    </div>
    """, unsafe_allow_html=True)


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
                    with st.expander(f'📁 {name}', expanded=False):
                        config_key = f"config_{name}"
                        sim_key = f"sim_{name}"
                        metrics_key = f"metrics_{name}"

                        #Configuration Button
                        config_button = st.button("⚙️ Config", key=config_key, use_container_width=True,
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
        metrics()

if __name__ == '__main__':
    main()

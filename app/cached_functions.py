from app.common import * 
from portfolio import Portfolio
from simulation import monte_carlo, historical
from metrics import calculate_metrics, SimulationAnalyzer
import pandas as pd
import streamlit as st


#========= CACHED PORTFOLIO FUNCTIONS =========

@st.cache_data(show_spinner=False)
def cached_get_data(tickers, lower_bound, upper_bound, period=None, start_date=None, end_date=None):
  """
  Cached version of get_data with hashable inputs.
  """
  portfolio = Portfolio(tickers, lower_bound, upper_bound)
  return portfolio.get_data(period, start_date, end_date)

@st.cache_data(show_spinner=False)
def cached_get_weights(tickers, lower_bound, upper_bound, period, start_date, end_date, type_weight, custom_weights=None):
    """
    Cached version of get_weights with hashable inputs.
    """
    portfolio = Portfolio(tickers, lower_bound, upper_bound)
    portfolio.get_data(period, start_date, end_date)
    return portfolio.get_weights(type_weight, custom_weights)


#========= CACHED SIMULATION FUNCTIONS =========

@st.cache_data(show_spinner=False)
def cached_monte_carlo(T, sims, weights, df_values, df_index, df_columns, regime, level, factor_stress=None, rand=None):
    """
    Cached Monte Carlo simulation with hashable inputs.
    """
    #Reconstruct DataFrame
    df = pd.DataFrame(data=df_values, index=df_index, columns=df_columns)
    return monte_carlo(T, sims, weights, df, regime, level, factor_stress, rand)

@st.cache_data(show_spinner=False)
def cached_historical(df_values, df_index, df_columns, crisis):
    """
    Cached historical simulation.
    """
    df = pd.DataFrame(data=df_values, index=df_index, columns=df_columns)
    return historical(df, crisis)


#========= CACHED METRICS FUNCTIONS =========

@st.cache_data(show_spinner=False)
def cached_calculate_metrics(weights, df_values, df_index, df_columns):
    """ 
    Cached version of calculate_metrics with hashable inputs.
    """
    df = pd.DataFrame(data=df_values, index=df_index, columns=df_columns)
    return calculate_metrics(weights, df)

@st.cache_data(show_spinner=False)
def cached_all_MC_metrics(mc_batches_dict, weights):
    """ 
    Cached version of all_MC_metrics computation.
    """
    analyzer = SimulationAnalyzer()
    analyzer.mc_batches = mc_batches_dict
    return analyzer.all_MC_metrics(weights)

import pandas as pd
import numpy as np
import yfinance as yf

def calculate_metrics(weights:list, df: pd.DataFrame):
  """
  Calculates annual portfolio volatilty, Sharpe Ratio, 95% VaR, Max Drawdown, and Beta.
  weights: list of each assets weight in the portfolio
  df: dataframe of the assets adjusted closes
  """
  try:
    if df is None or df.empty:
      raise ValueError("Price data is empty or unavailable. Make sure historical/simulated data is properly downloaded.")
      
    #Core calculations
    weights = np.array(weights)
    log_returns = np.log(df/df.shift()).dropna()
    expected_returns = log_returns.mean()*252
    cov_matrix = log_returns.cov()*252
    rf = 0.045
    port_returns = weights.T @ expected_returns
    port_returns_series = log_returns @ weights
    
    #Metrics
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe = (port_returns - rf)/port_vol
    VaR_95 = np.percentile(port_returns_series, 5)
    
    #Max Drawdown
    cum_returns = (1+port_returns_series).cumprod()
    cum_max = np.maximum.accumulate(cum_returns)
    drawdown = cum_returns/cum_max - 1
    mdd = drawdown.min() #drawdown values are negative
    
    #Beta
    market = yf.download("SPY", period='10y', progress=False, auto_adjust=False)["Adj Close"]
    market_returns = (np.log(market/market.shift()).dropna()).squeeze() #convert to series so that it works properly with port_returns_series
    if pd.api.types.is_integer_dtype(port_returns_series.index):
      #Simulated case: align by length
      market_returns = market_returns.tail(len(port_returns_series)).reset_index(drop=True)
      port_returns_series = port_returns_series.reset_index(drop=True)
    else:
      #Simulated historical case: align by date
      start_date = pd.to_datetime(port_returns_series.index[0])
      end_date = pd.to_datetime(port_returns_series.index[-1])
      if start_date and end_date not in market_returns.index: #first make sure that market data contains crisis event
        market = yf.download("SPY", start=start_date, end=end_date, progress=False, auto_adjust=False)["Adj Close"]
        market_returns = (np.log(market/market.shift()).dropna()).squeeze()

      #align by date for either simulated historical or historical case
      aligned_index = port_returns_series.index.intersection(market_returns.index)
      market_returns = market_returns.loc[aligned_index]
      port_returns_series = port_returns_series.loc[aligned_index]
    beta = port_returns_series.cov(market_returns) / market_returns.var()

    metrics = pd.DataFrame(data=[[port_vol, sharpe, VaR_95, mdd, beta]] ,columns=["Annual Volatilty", "Sharpe","95% VaR", "Max DD", "Beta"], index=[["Portfolio"]])
    return metrics

  except Exception as e:
    print(f"Error in calculate_metrics: {e}")
    return None

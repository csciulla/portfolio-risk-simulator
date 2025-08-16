import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from scipy.optimize import minimize

class Portfolio:
  def __init__(self, portfolio:list,  lower_bound:float, upper_bound:float):
    try:
      if lower_bound >= upper_bound:
        raise ValueError("Lower bound must be less than upper bound.")

      self.portfolio = portfolio
      self.lower_bound = lower_bound
      self.upper_bound = upper_bound
      self.portfolio_df = None
      self.weights = None
      self.normalized_df = None
      self.colors = None

    except Exception as e:
      return str(e)

  def get_data(self, period:str=None, start_date:str=None, end_date:str=None):
    """
    Returns a dataframe of the historical adjusted close prices of the assets.
    - Only one method of date input should be provided, either 'period' or 'start_date' and 'end_date'.
    - Length of time series should be large enough to handle metric calculations.

    Parameters:
    - period: yfinance time period (e.g., '3mo', '1y', '3y', '5y', 'ytd', 'max').
    - start_date: Start date of the time series. YYYY-MM-DD format.
    - end_date: End date of the time series. YYYY-MM-DD format.
    """
    try:
      if period and (start_date or end_date): #checks if both methods of date input are used
        raise ValueError("Provide either 'period' OR both 'start_date' and 'end_date' -- not both.")

      if period:
        period = period.strip()
        self.portfolio_df = yf.download(self.portfolio, period=period, progress=False, auto_adjust=False)["Adj Close"]
      elif start_date and end_date:
        start_date = start_date
        end_date = end_date
        self.portfolio_df = yf.download(self.portfolio, start=start_date, end=end_date, progress=False, auto_adjust=False)["Adj Close"]
      else:
        raise ValueError("You must provide either a 'period' or both 'start_date' and 'end_date'.")
      
      NaN_count = self.portfolio_df.isna().sum().sum()
      df_size = self.portfolio_df.size

      if self.portfolio_df.empty or self.portfolio_df is None:
        raise ValueError("Downloaded price data is empty or unavailable.")
      elif len(self.portfolio_df) <= 2:
        raise ValueError("Downloaded price data is too short.")
      elif len(self.portfolio_df) < 21: #average trading days in a month
        print("Warning: Limited price history may lead to unreliable metrics.")
        
      if NaN_count >= df_size//10:
        print("Warning: Some price data does not exist within the downloaded time period. Proceed with caution.")
      elif NaN_count == df_size:
        raise ValueError("All downloaded data is NaN. Check the ticker symbols and date range.")

      return self.portfolio_df, None

    except Exception as e:
      return None, str(e)
    

  def get_weights(self, type_weight:str, custom_weights:list[float]=None):
    """
    Returns a list of weights for the portfolio.

    Parameters:
    - type_weight: Input 'eq' for equal-weighted portfolio or 'opt' for optimized weights based on the Sharpe-Ratio.
                   Input 'custom' if you want to define your own weights; input them in the 'custom_weights' parameter.
    - custom: Input custom weights; optional
    """
    try:
      if self.portfolio_df is None or self.portfolio_df.empty:
        raise ValueError("The portfolio's price data is missing. Please properly run 'get_data' first.")
      elif len(self.portfolio_df) <= 2:
        raise ValueError("Downloaded price data is too short.")

      #Get log returns of each asset
      log_returns = np.log(self.portfolio_df/self.portfolio_df.shift()).dropna()

      #Calculate initial portfolio metrics
      tickers = list(self.portfolio)
      weights = np.repeat(1/len(self.portfolio), len(self.portfolio))
      expected_returns = log_returns.mean()*252
      cov_matrix = log_returns.cov()*252
      rf = 0.045

      #Set bounds and constraints for objective function
      bounds = [(self.lower_bound, self.upper_bound) for _ in range(len(self.portfolio))]
      constraints = {"type": "eq", "fun": lambda w: np.sum(w)-1}
      def neg_sharpe(w):
        port_ret = w.T @ expected_returns
        port_std = np.sqrt(w.T @ cov_matrix @ w)
        return -((port_ret - rf)/port_std)

      if type_weight.strip().lower() == "eq":
        self.weights = [float(i) for i in weights]
      elif type_weight.strip().lower() == "opt":
        optimized_weights = minimize(neg_sharpe, weights, method="SLSQP", bounds=bounds, constraints=constraints)
        self.weights = [round(float(i),4) for i in optimized_weights.x]
      elif type_weight.strip().lower() == "custom":
        if custom_weights is None:
          raise ValueError("Please enter weights if custom weights are desired.")
        elif sum(custom_weights) != 1.0:
          raise ValueError("Custom weights must sum up to 1.")
        elif len(custom_weights) != len(tickers):
          raise ValueError("Number of weights should match number of assets")
        else:
          self.weights = custom_weights
      else:
        raise ValueError("Select a valid input for 'type_weight' -- either 'eq' or 'opt'.")

      return self.weights, None

    except Exception as e:
      return None, str(e)
    

  def normalize_portfolio_data(self):
    """ 
    Normalizes the portfolio data to start at 100 for increased interpretability.
    """
    try:
      if self.portfolio_df.empty or self.portfolio_df is None:
        raise ValueError("Downloaded price data is empty or unavailable.")
      
      self.normalized_df  = self.portfolio_df.copy()
      tickers = list(self.normalized_df.columns)
      for ticker in tickers:
        first_price = self.normalized_df[ticker].iloc[0]
        self.normalized_df[ticker] = (self.normalized_df[ticker]/first_price) * 100

      return self.normalized_df, None
    
    except Exception as e:
      return None, str(e)
    
  def get_portfolio_colors(self):
    """
    Dynamically returns unique colors for each asset in the portfolio to be used in the UI plots.
    """
    try:
      if self.weights is None:
        raise ValueError("Portfolio weights are missing. Try portfolio configuration again.")

      tickers = list(self.portfolio)
      n_tickers = len(tickers)

      #Add a different shade of blue for each ticker dynamically
      hex_chars = ['1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
      self.colors = []
      for i in range(n_tickers):
        first_char = hex_chars[i % len(hex_chars)]
        second_char = hex_chars[(i // len(hex_chars)) % len(hex_chars)]
        self.colors.append(f'#00{first_char}{second_char}66')

      return self.colors, None
    
    except Exception as e:
      return None, str(e)


  def plot_pie(self):
    """
    Plots a pie chart of the portfolio weight allocation using Plotly.
    """
    try:
      if self.weights is None:
        raise ValueError("Portfolio weights are missing. Try portfolio configuration again.")

      tickers = list(self.portfolio)
      weights = self.weights
      colors = self.colors
      
      fig = go.Figure(data=[go.Pie(labels=tickers, values=weights, hole=0)])

      fig.update_layout(
        margin=dict(t=40, b=0, l=0, r=0)
      )
      fig.update_traces(
        textinfo='percent+label',
        marker=dict(colors=colors, line=dict(color='black', width=1))
      )
      return fig, None
      
    except Exception as e:
      return None, str(e)
    
  def plot_line(self):
    """
    Plots line chart of normalized portfolio data using Plotly.
    """
    try:
      if self.normalized_df is None:
        raise ValueError("No portfolio data available to plot. Run normalized_portfolio_data first.")
      if self.colors is None:
        raise ValueError("No valid colors are mapped to each asset. Run get_portfolio_colors first.")
      
      normalized_df = self.normalized_df
      colors = self.colors
      tickers = list(normalized_df.columns)

      fig = go.Figure()

      for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(
                x=normalized_df.index,
                y=normalized_df[ticker],
                mode='lines',
                name=ticker,
                line=dict(color=colors[i], width=2),
                hovertemplate=f'{ticker}: %{{y:.2f}}<extra></extra>'
            ))
        fig.update_layout(
          xaxis_title='Date',
          yaxis_title='Normalized Price',
          hovermode='x unified',
          legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
          ))
        
      return fig, None

    except Exception as e:
      return None, str(e)
    

  
if __name__ == "__main__":
    port = Portfolio(["AAPL", "XOM","JPM"], 0.0, 0.5)
    df = port.get_data(period="5y")
    print(df.head())
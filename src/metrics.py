import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(weights:list, df:pd.DataFrame):
  """
  Calculates the metrics and the percent contribution of risk for each stock in the portfolio.
  - Metrics returned: Annual portfolio volatilty, Sharpe Ratio, 95% VaR, 95% CVaR, Max Drawdown, and Beta.

  Parameters:
  - weights: List of each assets weight in the portfolio
  - df: Dataframe of the historical adjusted close prices of the assets or simulated returns via the monte_carlo function
  """
  try:
    if df is None or df.empty:
      raise ValueError("Price data is empty or unavailable. Make sure historical/simulated data is properly downloaded.")

    #Core calculations
    if df.iloc[0,0] < 1:
      log_returns = df
      weights = weights + [0.000]
    else:
      log_returns = np.log(df/df.shift()).dropna()

    tickers = list(df.columns)
    weights = np.array(weights)
    expected_returns = log_returns.mean()*252
    cov_matrix = log_returns.cov()*252
    rf = 0.045
    port_returns = weights.T @ expected_returns
    port_returns_series = log_returns @ weights
    port_returns_annualized = port_returns * 252

    #Metrics
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe = (port_returns - rf)/port_vol
    VaR_95 = np.percentile(port_returns_series, 5)
    CVaR_95 = port_returns_series[port_returns_series <= VaR_95].mean()

    #Max Drawdown
    cum_returns = (1+port_returns_series).cumprod()
    cum_max = np.maximum.accumulate(cum_returns)
    drawdown = cum_returns/cum_max - 1
    mdd = drawdown.min() #drawdown values are negative

    #Beta
    if pd.api.types.is_integer_dtype(port_returns_series.index):
      #Simulated case: align by length
      market_returns = df['SPY']
    else:
      market = yf.download("SPY", period='max', progress=False, auto_adjust=False)["Adj Close"]
      market_returns = (np.log(market/market.shift()).dropna()).squeeze() #convert to series so that it works properly with port_returns_series
      
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

    #Calculate PCR (Percent Contribution to Risk)
    PCRdict = {}
    if 'SPY' in tickers:
      tickers = tickers[:-1] #Remove 'SPY' 
      weights = weights[:-1]
    for i, ticker in enumerate(tickers):
      ticker_vol = np.std(log_returns[ticker]) * np.sqrt(252)
      ticker_corr = log_returns[ticker].corr(port_returns_series)
      MRC = ticker_vol*ticker_corr
      PCR = (weights[i]*MRC)/port_vol
      PCRdict[ticker] = (f"{PCR*100:.2f}%")
    PCRframe = pd.DataFrame(data=PCRdict, index=["PCR"])

    metrics = pd.DataFrame(data=[[port_vol, port_returns_annualized, sharpe, VaR_95, CVaR_95, mdd, beta]] ,columns=["Annual Volatilty", "Expected Return", "Sharpe","95% VaR", "95% CVaR", "Max DD", "Beta"], index=["Portfolio"])
    return (metrics, PCRframe), None

  except Exception as e:
    return None, str(e)
  
class SimulationAnalyzer:
    def __init__(self):
        self.batches = {}
        self.all_metrics_df = {}
        self.all_PCR_df = {}

    def add_simulation(self, label:str, sims_returns:dict[str, np.array]):
        """
        Add a new simulation batch under a user-defined label.

        Parameters:
        - label: Name of the simulation batch
        - sims_returns: The returns of the simulation derived from the 'monte_carlo' function 
        """
        try:
            if label in self.batches.keys():
                raise ValueError("Label already taken.")

            self.batches[label] = sims_returns
        
        except Exception as e:
            print(f"Error in add_simulation: {e}")
            return None

    def all_metrics(self, weights:list[float]):
        """
        Computes portfolio metrics and PCR values for each simulation path.

        Returns:
            - all_metrics_df: DataFrame where rows = simulations, cols = metrics.
            - all_PCR_df: DataFrame where rows = simulations, cols = tickers.
        """
        try:
            for label, sims_returns in self.batches.items():
                
                tickers = list(sims_returns.keys())
                num_sims = len(sims_returns[tickers[0]][0])

                all_metrics = []
                sims_index = []
                all_PCR = []
                for i in range(num_sims):
                    sims_index.append(f'Simulation {i+1}')
                    sim_mth_df = pd.DataFrame({ticker: sims_returns[ticker][:, i] for ticker in tickers})

                    metrics_df, PCR_df = calculate_metrics(weights, sim_mth_df)
                    all_metrics.append(metrics_df.loc['Portfolio'])
                    all_PCR.append(PCR_df.loc['PCR'])

                self.all_metrics_df[label] = pd.DataFrame(data=all_metrics, index=sims_index)
                self.all_PCR_df[label] = pd.DataFrame(data=all_PCR, index=sims_index)

            return self.all_metrics_df, self.all_PCR_df
        
        except Exception as e:
            print(f"Error in all_metrics function: {e}")
            return None
        
    def visualize_metrics(self):

        try:
            metrics = list(next(iter(self.all_metrics_df.values())).columns)
            num_metrics = len(metrics)

            fig, axes = plt.subplots(1, num_metrics, figsize=(4*num_metrics, 5), constrained_layout=True)
            for idx, metric in enumerate(metrics):
                ax=axes[idx]
                all_vals = pd.concat([df[metric] for df in self.all_metrics_df.values()])
                x_min, x_max = all_vals.min(), all_vals.max()
                
                for df in self.all_metrics_df.values():
                    sns.kdeplot(df[metric], ax=ax)
                ax.set_title(f"KDE of {metric}")
                ax.set_xlabel(metric)
                ax.set_ylabel("Density")
                ax.grid(True)
                ax.set_xlim(x_min, x_max)

            fig.suptitle("KDE Comparison of Portfolio Metrics Across Simulations", fontsize=16)
            fig.legend(labels=self.all_metrics_df.keys())
            plt.show()
               
                
        except Exception as e:
            print(f"Error in visualize metrics: {e}")
            return None
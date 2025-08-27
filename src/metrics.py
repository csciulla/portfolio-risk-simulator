import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity

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

    metrics = pd.DataFrame(data=[[port_vol, port_returns_annualized, sharpe, VaR_95, CVaR_95, mdd, beta]] ,columns=["Annual Volatilty", "Expected Return", "Sharpe","95% VaR", "95% CVaR", "Max Drawdown", "Beta"], index=["Portfolio"])
    return (metrics, PCRframe, cum_returns), None

  except Exception as e:
    return None, str(e)
  
  
class SimulationAnalyzer:
    def __init__(self):
        self.mc_batches = {}
        self.hist_batches = {}
        self.all_mc_metrics = {}
        self.all_mc_PCR = {}
        self.all_mc_cum_returns = {}

    def add_simulation(self, label:str, type:str, sims_returns:dict[str, np.array] | pd.DataFrame):
        """
        Add a new simulation batch under a user-defined label.

        Parameters:
        - label: Name of the simulation batch
        - type: Type of simulation -- 'Monte Carlo' or 'Historical Replay'
        - sims_returns: The returns of the simulation derived from the 'monte_carlo' for 'historical' function
        """
        try:
            if label in self.mc_batches.keys() or label in self.hist_batches.keys():
                raise ValueError("Label already taken.")

            if type == 'Monte Carlo':
              self.mc_batches[label] = sims_returns
            elif type == 'Historical Replay':
               self.hist_batches[label] = sims_returns
            else:
               raise ValueError("Insert valid type. Either 'monte_carlo' or 'historical'.")
            
            return True, None
        
        except Exception as e:
            return None, str(e)


    def all_metrics(self, weights:list[float]):
        """
        Computes portfolio metrics and PCR values for each simulation path.
        Only for monte carlo scenarios.
        """
        try:
            for label, sims_returns in self.mc_batches.items():
                
                tickers = list(sims_returns.keys())
                num_sims = len(sims_returns[tickers[0]][0])

                all_metrics = []
                sims_index = []
                all_PCR = []
                all_cum_returns = []
                for i in range(num_sims):
                    sims_index.append(f'Simulation {i+1}')
                    sim_mth_df = pd.DataFrame({ticker: sims_returns[ticker][:, i] for ticker in tickers})

                    results, error = calculate_metrics(weights, sim_mth_df)
                    if error:
                       raise ValueError(f"Error in path {i} metrics calculation: {error}")
                    
                    metrics_df = results[0]
                    PCR_df = results[1]
                    cum_returns = results[-1]
                    all_metrics.append(metrics_df.loc['Portfolio'])
                    all_PCR.append(PCR_df.loc['PCR'])
                    all_cum_returns.append(cum_returns)

                self.all_mc_metrics[label] = pd.DataFrame(data=all_metrics, index=sims_index)
                self.all_mc_PCR[label] = pd.DataFrame(data=all_PCR, index=sims_index)
                self.all_mc_cum_returns[label] = pd.DataFrame(data=all_cum_returns, index=sims_index)

            return (self.all_mc_metrics, self.all_mc_PCR, self.all_mc_cum_returns), None
        
        except Exception as e:
            return None, str(e)
        
    def get_display_path(self, label:str, type_path:str, use_metrics=None):
        """ 
        Returns the metrics of the desired path from the all monte carlo metrics dataframe.

        Parameters:
        - type_path: Type of path to return -- 'Best', 'Worst', or 'Representative'
        - use_metrics: list of column names from self.all_mc_metrics[0] to fit KDE on
        """
        try:    
            if type_path != 'Representative':
                sharpes = self.all_mc_metrics[label]['Sharpe'].values

                if type_path == 'Best':
                    idx = np.argmax(sharpes)
                
                elif type_path == 'Worst':
                    idx = np.argmin(sharpes)                  
                
            elif type_path == 'Representative':
                if use_metrics is None:
                    use_metrics = ['Annual Volatilty', 'Sharpe', '95% VaR', 'Max Drawdown']

                metrics_df = self.all_mc_metrics[label]

                #Build joint dataset
                X = metrics_df.loc[:, use_metrics].values
                kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)

                #Score each path
                log_probs = kde.score_samples(X)
                idx = np.argmax(log_probs)
            
            else:
                raise ValueError("Invaild type_path input. Must be either 'Best', 'Worst', or 'Representative'.")
            
            #Extract results
            metrics = self.all_mc_metrics[label].iloc[idx].copy()
            PCR = self.all_mc_PCR[label].iloc[idx].copy()
            cum_returns = self.all_mc_cum_returns[label].iloc[idx].copy()

            return (metrics, PCR, cum_returns), None
            
        except Exception as e:
            return None, str(e)
        
        
    def get_confidence_intervals(self, confidence_level:float=0.95):
        """
        Calculate confidence intervals for each metric across all simulations.

        Parameters:
        - confidence_level: The desired confidence level (default is 0.95 for 95% CI)

        Returns:
            - ci_dict: Dictionary with labels as keys and DataFrames of confidence intervals as values.
        """
        try:
            ci_dict = {}
            alpha = 1 - confidence_level
            lower_bound = alpha / 2
            upper_bound = 1 - (alpha / 2)

            for label, metrics_df in self.all_mc_metrics.items():
                ci_data = {}
                for metric in metrics_df.columns:
                    ci_lower = metrics_df[metric].quantile(lower_bound)
                    ci_upper = metrics_df[metric].quantile(upper_bound)
                    ci_data[metric] = (ci_lower, ci_upper)
                ci_dict[label] = pd.DataFrame(ci_data, index=['Lower CI', 'Upper CI']).T

            return ci_dict, None
        
        except Exception as e:
            return None, str(e)
        

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
            return None, str(e)
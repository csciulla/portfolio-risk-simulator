import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde


def calculate_metrics(weights:list, df:pd.DataFrame):
  """
  Calculates the metrics and the percent contribution of risk for each stock in the portfolio.
  - Metrics returned: Annual portfolio volatility, Sharpe Ratio, 95% VaR, 95% CVaR, Max Drawdown, and Beta.

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
    else:
      if list(df.columns)[-1] != 'SPY': #For historical case -- not historical replay
        market_prices = yf.download('SPY', start=df.index[0], end=df.index[-1], progress=False, auto_adjust=False)['Adj Close']
        df = pd.concat([df, market_prices], axis=1)

      log_returns = np.log(df/df.shift()).dropna()

    tickers = list(df.columns)
    weights = weights + [0.0]
    weights = np.array(weights)
    expected_returns = log_returns.mean()*252
    cov_matrix = log_returns.cov()*252
    rf = 0.045
    port_returns = weights.T @ expected_returns
    port_returns_series = log_returns @ weights

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
    market_returns = log_returns['SPY']
    beta = port_returns_series.cov(market_returns) / market_returns.var()

    #Calculate PCR (Percent Contribution to Risk)
    PCRdict = {}
    if tickers[-1] == 'SPY':
      tickers = tickers[:-1] #Remove 'SPY' 
      weights = weights[:-1]
    for i, ticker in enumerate(tickers):
      ticker_vol = np.std(log_returns[ticker]) * np.sqrt(252)
      ticker_corr = log_returns[ticker].corr(port_returns_series)
      MRC = ticker_vol*ticker_corr
      PCR = (weights[i]*MRC)/port_vol
      PCRdict[ticker] = PCR*100
    PCRframe = pd.DataFrame(data=PCRdict, index=["PCR"])

    metrics = pd.DataFrame(data=[[port_vol, port_returns, sharpe, VaR_95, CVaR_95, mdd, beta]] ,columns=["Annual Volatility", "Expected Return", "Sharpe Ratio","95% VaR", "95% CVaR", "Max Drawdown", "Beta"], index=["Portfolio"])
    return (metrics, PCRframe, cum_returns), None

  except Exception as e:
    return None, str(e)


def plot_cumulative_returns(inital_value:int, view:str, cum_df:pd.DataFrame, current_scenario:str, current_path:str=None):
    """ 
    Plots cumulative return line chart using Plotly.

    Parameters:
    - inital_value: Integer of starting portfolio value in USD
    - view: String of view mode -- 'Detailed View' for scenario paths or 'Compare Scenarios' for scenario comparison
    - cum_df: Dataframe of cumulative returns of cumulative returns of representative, best, and worst paths (Detailed View) 
                or Dataframe of representative paths of all scenarios (Compare Scenarios)

    - current_scenario: String of the currently selected scenario
    - current_path: String of the currently selected path ('Representative', 'Best', 'Worst'); required if view is 'Detailed View'
    """
    try:
        fig = go.Figure()
        
        #Detailed View: plots representative, best, and worst paths of the current scenario (Only for Monte Carlo)
        if view == 'Detailed View' and current_path is not None:
            path_colors = ['#1d5fa2', '#1563ad', '#08519c']
            labels = list(cum_df.columns)
            for i, label in enumerate(labels):

                #Opacity based on if current path or not
                opacity = 1.0 if label == current_path else 0.3
                line_width = 3 if label == current_path else 2

                fig.add_trace(go.Scatter(
                x=cum_df.index,
                y=cum_df[label]*inital_value,
                mode='lines',
                name=f'{current_scenario} - {label}',
                line=dict(color=path_colors[i], width=line_width),
                opacity=opacity,
                hovertemplate=f'{label}: %{{y:.2f}}<extra></extra>'
                ))

                fig.update_layout(
                title=f'Detailed View Across Paths',
                xaxis_title='Number of Days',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                margin=dict(t=60, b=0, l=0, r=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ))
        
            return fig, None
        
        #Compare Scenarios: plots representative paths of all scenarios for comparison
        elif view == 'Compare Scenarios' and current_path is None:
            scenario_colors = px.colors.qualitative.Plotly
            labels = list(cum_df.columns)
            for i, label in enumerate(labels):
                line_width = 3 if label == current_scenario else 2
                opacity = 1.0 if label == current_scenario else 0.5

                fig.add_trace(go.Scatter(
                    x=cum_df.index,
                    y=cum_df[label]*inital_value,
                    mode='lines',
                    name=label,
                    line=dict(color=scenario_colors[i], width=line_width),
                    opacity=opacity,
                    hovertemplate=f'{label}: %{{y:.2f}}<extra></extra>'
                ))

                fig.update_layout(
                title='Across Scenarios',
                xaxis_title='Number of Days',
                yaxis_title='Portfolio Value ($)',
                hovermode='x unified',
                margin=dict(t=60, b=0, l=0, r=0),
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

def plot_PCR(sim_PCR_df: pd.DataFrame, baseline_PCR_df: pd.DataFrame):
    """
    Plots bar chart of simulated PCR overlaid with baseline PCR.
    Plots average PCR if Monte Carlo scenario due to low variance; single PCR if Historical Replay.

    Parameters:
    - sim_PCR_df: Dataframe of PCR from simulated scenario; average of all paths if Monte Carlo secenario
    - baseline_PCR_df: Dataframe of PCR from the original data the simulations are based on
    """
    try:
        fig = go.Figure()
        sim_PCR = sim_PCR_df.iloc[0].astype(float)
        tickers = sim_PCR.index
        fig.add_trace(go.Bar(
            x=tickers,
            y=sim_PCR.values,
            name="Simulated Avg PCR" if sim_PCR.name[0] == 'Avg PCR' else 'Simulated PCR',
            marker_color='#08306b',
            opacity=0.7,
        ))

        #Baseline PCR overlay
        baseline_PCR = baseline_PCR_df.iloc[0].astype(float)
        fig.add_trace(go.Bar(
            x=baseline_PCR.index,
            y=baseline_PCR.values,
            name="Baseline PCR",
            marker_color='#6baed6',
            opacity=0.5
        ))

        fig.update_layout(
            xaxis_title="Tickers",
            yaxis_title="Percent Contribution of Risk (%)",
            barmode='overlay',
            height=548,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig, None

    except Exception as e:
        return None, str(e)
  
  
class SimulationAnalyzer:
    def __init__(self):
        self.mc_batches = {}
        self.hist_batches = {}

        self.all_mc_metrics = {}
        self.all_mc_PCR = {}
        self.all_mc_cum_returns = {}

        self.combined_metrics = {}
        self.combined_PCR = {}
        self.combined_cum_returns = {}

    def add_simulation(self, label:str, type:str, sim_result:dict[str, np.array] | pd.DataFrame):
        """
        Add a new simulation batch under a user-defined label.

        Parameters:
        - label: Name of the simulation batch
        - type: Type of simulation -- 'Monte Carlo' or 'Historical Replay'
        - sims_result: The result of the simulation derived from the 'monte_carlo' for 'historical' function
        """
        try:
            if label in self.mc_batches.keys() or label in self.hist_batches.keys():
                raise ValueError("Label already taken.")

            if type == 'Monte Carlo':
              self.mc_batches[label] = sim_result
            elif type == 'Historical Replay':
               self.hist_batches[label] = sim_result
            else:
               raise ValueError("Insert valid type. Either 'monte_carlo' or 'historical'.")
            
            return self.hist_batches, None
        
        except Exception as e:
            return None, str(e)


    def all_MC_metrics(self, weights:list[float]):
        """
        Computes portfolio metrics, PCR values, and cumulative returns for each simulation path.
        Only for Monte Carlo simulations.
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
        Returns the metrics of the desired path from the all Monte Carlo metrics dataframe.
        Only for Monte Carlo simulations.

        Parameters:
        - type_path: Type of path to return -- 'Best', 'Worst', or 'Representative'
        - use_metrics: list of column names from self.all_mc_metrics[0] to fit KDE on
        """
        try:    
            if type_path != 'Representative':
                sharpes = self.all_mc_metrics[label]['Sharpe Ratio'].values

                if type_path == 'Best':
                    idx = np.argmax(sharpes)
                
                elif type_path == 'Worst':
                    idx = np.argmin(sharpes)                  
                
            elif type_path == 'Representative':
                if use_metrics is None:
                    use_metrics = ['Annual Volatility', 'Expected Return', 'Sharpe Ratio', '95% VaR', 'Max Drawdown']

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


    def get_combined_metrics(self, weights:list[float]):
        """ 
        Combines Monte Carlo and Historical Replay metrics into a single dictionary.

        Parameters:
        - weights: List of portfolio weights
        """
        try:
            #Add Monte Carlo scenarios (already computed)
            self.combined_metrics.update(self.all_mc_metrics)
            self.combined_PCR.update(self.all_mc_PCR)
            self.combined_cum_returns.update(self.all_mc_cum_returns)

            #Get Historical Replay Metrics
            for label, hist_data in self.hist_batches.items():
                results, error = calculate_metrics(weights, hist_data)
                if error:
                    raise ValueError(f"Error calculating metrics for {label}: {error}")
                
                metrics_df, PCR_df, cum_returns = results

                self.combined_metrics[label] = metrics_df
                self.combined_PCR[label] = PCR_df
                self.combined_cum_returns[label] = pd.DataFrame([cum_returns], index=[label])
            
            return (self.combined_metrics, self.combined_PCR, self.combined_cum_returns), None
        
        except Exception as e:
            return None, str(e)


    def visualize_metrics(self, metric_name):
        """
        Create KDE plot of a single portfolio metric across ALL scenarios (Monte Carlo + Historical Replay).
        Requires weights to be stored as an attribute.
        
        Parameters:
        - metric_name: str, name of the metric to visualize (e.g., 'Sharpe Ratio', 'Annual Volatility', etc.)
        """
        try:
            if not hasattr(self, 'weights'):
                raise ValueError("Weights must be stored as class attribute to visualize all scenarios")
                
            combined_data, error = self.get_combined_metrics(self.weights)
            if error:
                raise ValueError(f"Error getting combined metrics: {error}")
                
            combined_metrics, _, _ = combined_data
            
            # Validate metric_name exists
            available_metrics = list(next(iter(combined_metrics.values())).columns)
            if metric_name not in available_metrics:
                raise ValueError(f"Metric '{metric_name}' not found. Available metrics: {available_metrics}")
            
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            
            #Get overall range for the metric across all scenarios
            all_vals = pd.concat([df[metric_name] for df in combined_metrics.values()])
            x_min, x_max = all_vals.min(), all_vals.max()
            
            padding = (x_max - x_min) * 0.05
            x_range = np.linspace(x_min - padding, x_max + padding, 100)

            for i, (scenario_name, df) in enumerate(combined_metrics.items()):
                values = df[metric_name].values

                if len(values) > 1: #Monte Carlo case
                    kde = gaussian_kde(values)
                    density = kde(x_range)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range,
                            y=density,
                            mode='lines',
                            name=scenario_name,
                            line=dict(
                                color=colors[i],
                                width=2
                            ),
                            hovertemplate=f'<b>{scenario_name}</b><br>' +
                                        f'{metric_name}: %{{x:.4f}}<br>' +
                                        'Density: %{y:.4f}<extra></extra>'
                        )
                    )
                    
                else: #Historical Replay case - only vertical line with legend entry
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode='lines',
                            name=scenario_name,
                            line=dict(
                                color=colors[i],
                                width=3,
                                dash='dash'
                            ),
                            showlegend=True,
                            hoverinfo='skip'
                        )
                    )
                    
                    #Add vertical line
                    fig.add_vline(
                        x=values[0],
                        line_dash="dash",
                        line_color=colors[i],
                        line_width=3,
                        opacity=0.8
                    )

            fig.update_layout(
                title={
                    'text': f'KDE Distribution of {metric_name} Across All Scenarios',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 18}
                },
                xaxis_title=metric_name,
                yaxis_title="Density",
                height=500,
                width=800,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                hovermode='closest',
                template='plotly_white',
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)'
                )
            )
            
            return fig, None
        
        except Exception as e:
            return None, str(e)
        
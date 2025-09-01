import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from hmmlearn.hmm import GaussianHMM

def monte_carlo(T:int, sims:int, weights:list, df:pd.DataFrame, regime:str, level:str, factor_stress:list=None, rand:bool=None ):
  """
  Returns simulated portfolio returns using Monte Carlo Simulation.

  Parameters:
  - T: Number of days in each simulation.
  - sims: Number of simulations.
  - weights: List of asset weights.
  - df: Dataframe of the historical adjusted close prices of the assets.
  - regime: Volatility environment determined by the Hidden Markov Model
  - level: Crisis severity multiplier applied to the volatility regime
  - factor_stress: shifts the simulations based on the expected returns stressed on factors; optional.
  - rand: input the boolean 'True' to return a random simulation, otherwise ignore

    regime options: 'Low', 'Medium', 'High'
    level options: 'Mild', 'Moderate', 'Severe', 'Tail Risk', 'Regulatory'
  """
  try:
    if T <= 2:
      raise ValueError("The length of each simulated path is too short.")
    elif T < 21:
      print("Warning: Limited price data may lead to unreliable metrics.")

    #Intialize dictionary to store simulated paths of T days for each ticker
    tickers = list(df.columns) + ['SPY']
    weights = weights + [0.000]
    sims_returns = {ticker: np.full(shape=(T, sims), fill_value=0.0) for ticker in tickers}
    
    #Correspond regime with scaling factor
    regime = regime.strip().capitalize()
    level = level.strip().capitalize()
    factorDict = {"Mild": 1.0,
                  "Moderate": 1.15,
                  "Severe": 1.35,
                  "Tail risk": 1.6,
                  "Regulatory": 1.8}
    scaling_factor = factorDict[level]

    #Calculate log returns and align with market returns
    start_date = pd.to_datetime(df.index[0])
    end_date = pd.to_datetime(df.index[-1])
    market = yf.download('SPY', start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
    market_returns = np.log(market/market.shift()).dropna()
    log_returns = np.log(df/df.shift()).dropna()
    aligned_index = log_returns.index.intersection(market_returns.index)
    market_returns = market_returns.loc[aligned_index]
    log_returns = log_returns.loc[aligned_index]
    log_returns['SPY'] = market_returns

    #Create mean matrix 
    if factor_stress is not None:
      meanM = np.full(shape=(T, len(tickers)), fill_value=factor_stress)
    else:
      expected_return = log_returns.mean()
      meanM = np.full(shape=(T, len(tickers)), fill_value=expected_return)

    #Initalize HMM
    port_returns = (log_returns @ weights).values.reshape(-1,1) #HMM requires 2D array
    historical_port_vol = np.std(port_returns)
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(port_returns)

    #Gather the volatility regimes established by the HMM and correspond them with their respective state
    vol_states = ["Low","Medium","High"]
    vol_regimes = np.sqrt([var[0][0] for var in model.covars_])
    vol_regimes = np.sort(vol_regimes)
    vol_dict = {state: vol for state, vol in zip(vol_states, vol_regimes)}

    #Calculate the scale factor needed for the historical data to reach the desired volatility and then apply it to L
    desired_vol = vol_dict[regime]*scaling_factor
    vol_scale_factor = desired_vol / historical_port_vol
    cov_matrix = log_returns.cov()* (vol_scale_factor**2)
    L = np.linalg.cholesky(cov_matrix)

    #Generate paths
    for m in range(sims):
      Z = np.random.normal(size=(T, len(tickers)))
      dailyReturns = meanM + Z @ L.T
      for i, ticker in enumerate(tickers):
        sims_returns[ticker][:,m] = dailyReturns[:,i]

    #Get a random path
    if rand:
      random_int = np.random.randint(0,sims)
      random_sims_returns = {ticker: sims_returns[ticker][:,random_int] for ticker in tickers}
      random_sims_df = pd.DataFrame(random_sims_returns)
      return random_sims_df, None
    elif rand != None:
      raise ValueError("Invaild input for 'rand'. Input the string 'yes' to return a random path, otherwise ignore.")
    else:
        return sims_returns, None

  except Exception as e:
    return None, str(e)
  
  
def historical(df:pd.DataFrame, crisis:str):
  """
  Returns the prices of your portfolio if during a historical crisis event.
  
  Parameters:
  - df: Dataframe of the historical adjusted close prices of the assets.
  - crisis: String of the event you want to simulate.

    Crisis Options:
    - "DOT-COM" -- The Dot-Com bubble
    - "2008 GFC" -- 2008 Global Financial Crisis
    - "2011 Euro" -- 2011 Eurozone Crisis
    - "COVID" -- COVID-19 Pandemic
    - "2022 Inf" -- 2022 Inflation Crash
  """
  try:
    crisis_periods = {"DOT-COM": ("2000-03-01", "2002-10-01"),
                      "2008 GFC": ("2007-10-01", "2009-03-01"),
                      "2011 Euro": ("2011-07-01", "2011-12-01"),
                      "COVID": ("2020-02-14", "2020-04-15"),
                      "2022 Inf": ("2022-01-01", "2022-10-01")
                      }
    crisis = crisis.strip()
    if crisis not in crisis_periods.keys():
      raise ValueError("Input a valid crisis event.")

    if list(df.columns)[-1] != 'SPY':
      market_data = yf.download('SPY', start=df.index[0], end=df.index[-1], progress=False, auto_adjust=False)['Adj Close']
      df = pd.concat([df, market_data], axis=1)

    tickers = list(df.columns)

    start_date = pd.to_datetime(crisis_periods[crisis][0])
    end_date = pd.to_datetime(crisis_periods[crisis][1])

    if start_date not in df.index: #check if crisis event does not exist in existing df
      dfcrisis = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)["Adj Close"]
    else:
      dfcrisis = df.loc[start_date:end_date]

    for ticker in tickers:
      if dfcrisis[ticker].isna().sum() >= len(dfcrisis[ticker])//3: #checks if any ticker reaches NA threshold
        raise ValueError(f"{ticker} price data does not exist for crisis period.")

    if df.iloc[-1].isna().any():
      df = df.iloc[:-1]
      
    last_price = df.iloc[-1]
    crisisReturns = np.log(dfcrisis/dfcrisis.shift()).dropna()
    cumReturns = (1+crisisReturns).cumprod()
    crisisPrices = last_price.mul(cumReturns)
    crisisPrices.index = range(len(crisisPrices))
    return crisisPrices, None

  except Exception as e:
    return None, str(e)
  
class FactorStress:
    def __init__(self, portfolio_df:pd.DataFrame):     
        self.portfolio_df = portfolio_df
        self.log_returns = None
        self.final_factors = None
        self.factors_df = None
        self.rf_df = None
        self.classifications = None
        self.results = None
        
    def process_factors(self, factors:list[str]):
        """
        Input a list of factors that you want to contribute to the classification of each asset.
        Cleans factor CSVs to align with portfolio data and only include desired factors.
        Stores aligned factor data, portfolio log returns, and risk-free rate data in self variables.

        Parameters:
        - factors: A list of strings corresponding to the factors.

        factor options:
        - 'FF3' = Fama-French 3-Research Factors: ['Mkt-RF', 'SMB', 'HML']
        - 'FF5' = Fama-French 5-Research Factors: ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        - 'Mkt-RF' = Market Returns - Risk-Free-Rate: Market Risk Premium.
        - 'SMB' = Small Minus Big: Returns of small-cap stocks minus large-cap stocks; measures the size anomaly - small-cap tend to outperform.
        - 'HML' = High Minus Low: Returns of high B/M (value) stocks minus low B/M (growth) stocks; measures the value anomaly - value stocks tend to outperform.
        - 'RMW' = Robust Minus Weak: Returns of firms with robust profitability minus those with weak profitability; measures profitability factor - more robust firms tend to earn higher returns.
        - 'CMA' = Conservative Minus Aggressive: Returns of firms with conservative investment minus aggressive policies; captures investment factor - conservative tend to perform better.
        - 'Mom' = Momentum: Returns of stocks with high prior returns minus those with low prior returns; captures the momentum effect where past winners tend to continue outperforming in the short term.

        """
        try:
            factor_map = {
                'FF3': ['Mkt-RF', 'SMB', 'HML'],
                'FF5': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
            }

            extended = []
            for factor in factors:
                if factor in factor_map.keys():
                    extended.extend(factor_map[factor])
                else:
                    extended.append(factor)
            self.final_factors = list(set(extended))

            #Read and clean factor CSVs
            FFdf = pd.read_csv('./data/F-F_Research_Data_5_Factors_2x3_daily.csv', skiprows=3, index_col=0).iloc[:-1]
            FFdf.index = pd.to_datetime(FFdf.index)
            MOMdf = pd.read_csv('./data/F-F_Momentum_Factor_daily.csv', skiprows=13, index_col=0).iloc[:-1]
            MOMdf.index = pd.to_datetime(MOMdf.index)

            #Align both factor CSVs and only grab necessary factors
            factor_align = MOMdf.index.intersection(FFdf.index)
            FFdf = FFdf.loc[factor_align]
            MOMdf = MOMdf.loc[factor_align]
            full_factors = pd.concat([FFdf, MOMdf], axis=1)
            rf = full_factors['RF'].copy()
            full_factors = full_factors[self.final_factors]
        
            #Align historical asset returns with joint factor df
            log_returns = np.log(self.portfolio_df/self.portfolio_df.shift()).dropna()
            aligned_index = log_returns.index.intersection(full_factors.index) 
            self.log_returns = log_returns.loc[aligned_index]
            self.factors_df = full_factors.loc[aligned_index]/100
            self.rf_df = rf.loc[aligned_index]/100

            return self.factors_df, None

        except Exception as e:
            return None, str(e)


    def classify_factors(self):
        """ 
        Conducts classification of assets through OLS based on the factors desired.
        """
        try:
            #Fit OLS on each ticker and store neccesary values
            tickers = list(self.portfolio_df.columns)
            self.results = {}
            classifications = {ticker: [] for ticker in tickers}
            for ticker in tickers:
                excess_returns = self.log_returns[ticker] - self.rf_df
                X = sm.add_constant(self.factors_df)
                y = excess_returns
                
                model = sm.OLS(y, X).fit()
                self.results[ticker] = {
                    'alpha': model.params['const'],
                    'betas': model.params.drop('const').to_dict(),
                    'r-squared': model.rsquared
                }
                #Stock Classification
                betas = self.results[ticker]['betas']

                for factor in self.final_factors:
                    #Mkt-RF
                    if factor == 'Mkt-RF': 
                        if betas[factor] > 1.1:
                            classifications[ticker].append('High-Beta')
                        elif betas[factor] < 0.9:
                            classifications[ticker].append('Low-Beta')
                        else:
                            classifications[ticker].append('Normal-Beta')

                    #SMB
                    if factor == 'SMB':
                        if betas[factor] > 0.3:
                            classifications[ticker].append('Small-Cap')
                        elif betas[factor] < -0.3:
                            classifications[ticker].append('Large-Cap')
                        else:
                            classifications[ticker].append('Mid-Cap')

                    #HML
                    if factor == 'HML':
                        if betas[factor] > 0.3:
                            classifications[ticker].append('Value')
                        elif betas[factor] < -0.3:
                            classifications[ticker].append('Growth')
                        else:
                            classifications[ticker].append('Blend')

                    #RMW
                    if factor == 'RMW': 
                        if betas[factor] > 0.2:
                            classifications[ticker].append("High-Quality")
                        elif betas[factor] < -0.2:
                            classifications[ticker].append('Low-Quality')
                        else:
                            classifications[ticker].append('Normal-Quality')

                    #CMA
                    if factor == 'CMA': 
                        if betas[factor] > 0.2:
                            classifications[ticker].append('Conservative')
                        elif betas[factor] < -0.2:
                            classifications[ticker].append('Agressive')
                        else:
                            classifications[ticker].append('Moderate')

                    #MOM
                    if factor == 'Mom':
                        if betas[factor] > 0.1:
                            classifications[ticker].append('High-Momentum')
                        elif betas[factor] < -0.1:
                            classifications[ticker].append('Low-Momentum')
                        else:
                            classifications[ticker].append('Mid-Momentum')

            classifications_df = pd.DataFrame.from_dict(data=classifications, orient='index', columns=self.final_factors)
            self.classifications = classifications_df
            return classifications_df, None
        
        except Exception as e:
            return None, str(e)
    
    def stress_means(self, shocks:dict):
        """ 
        Returns a list of the stressed expected returns for each asset based on the factors shocked.

        Parameters:
        - shocks: A dictionary of factor, shock pairs denoting the percentage increase or decrease you want each factor to be shocked by.
                  = Example: {'SMB': 0.2} shocks the size premium by 20%; tests how your portfolio performs when small-cap returns surge.
                  
                  = If 'FF3' or 'FF5' is entered in factors, enter a list within shocks that corresponds to each of factors or enter a float to apply a unform shock to all factors within 'FF3' or 'FF5'. 
                  = Example: {'FF3':[0.1, 0.2, 0.3], 'Mom':0.4]} or {'FF3':0.1, 'Mom':0.4} to apply uniform shock to 'FF3'
        """
        try:
            for factor in shocks.keys():
                if factor not in (self.final_factors + ['FF3','FF5']):
                    raise ValueError("You cant not shock a factor that was not included in the multi-factor-model.")
                
            #Standardize shock dict
            for factor in self.final_factors:
                if factor not in shocks.keys():
                    shocks[factor] = 0

            #Handle 'FF3' and 'FF5' shocks
            warning = None
            FF3_check = FF5_check = False
            if 'FF3' in shocks.keys():
                FF3_check = True
                FF3_value = shocks.pop('FF3')

                if isinstance(FF3_value, list): #Unique shocks per factor in 'FF3'
                    if len(FF3_value) == 3:
                        shocks.update({
                            'Mkt-RF': FF3_value[0],
                            'SMB': FF3_value[1],
                            'HML': FF3_value[2]
                            })
                    else:
                        raise ValueError("Invalid shock length for 'FF3'. Either input a float for a uniform shock or a list of independent shocks for each factor.")
                else: #Uniform shock to all factors in 'FF3'
                    shocks.update({
                        'Mkt-RF': FF3_value,
                        'SMB': FF3_value,
                        'HML': FF3_value
                        })
            if 'FF5' in shocks.keys():
                FF5_check = True
                FF5_value = shocks.pop('FF5')

                if isinstance(FF5_value, list): #Unique shocks per factor in 'FF5'
                    if len(FF5_value) == 5:
                        shocks.update({
                            'Mkt-RF': FF5_value[0],
                            'SMB': FF5_value[1],
                            'HML': FF5_value[2],
                            'RMW': FF5_value[3],
                            'CMA': FF5_value[4]
                        })
                    else:
                        raise ValueError("Invalid shock length for 'FF5'. Either input a float for a uniform shock or a list of independent shocks for each factor.")
                else: #Uniform shock to all factors in 'FF5'
                   shocks.update({
                        'Mkt-RF': FF5_value,
                        'SMB': FF5_value,
                        'HML': FF5_value,
                        'RMW': FF5_value,
                        'CMA': FF5_value
                        })
            if FF3_check and FF5_check:
                warning = "Both 'FF3' and 'FF5' have been selected. FF5 factors will overwrite overlapping FF3 factors."
                print(shocks)

            #Compute stressed mean returns
            stressed_means = {}
            factor_means = self.factors_df.mean()
            tickers = list(self.portfolio_df.columns)
            for ticker in tickers:
                alpha = self.results[ticker]['alpha']
                betas = self.results[ticker]['betas']
                stressed_mean = alpha
                for factor in self.final_factors:
                    stressed_mean += betas[factor]*factor_means[factor]*(1+shocks[factor])
                stressed_means[ticker] = stressed_mean
            stressed_means['SPY'] = 0.0
            stressed_means_list = list(map(float, list(stressed_means.values())))

            return stressed_means_list, None, warning

        except Exception as e:
            return None, str(e), None
        
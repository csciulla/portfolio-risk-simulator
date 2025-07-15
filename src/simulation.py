import pandas as pd
import numpy as np
import yfinance as yf

def monte_carlo(T:int, sims:int, vol_mult:float, df:pd.DataFrame, rand:str=None):
  """
  Returns simulated prices for each ticker in the portfolio using Monte Carlo Simulation.

  T: number of days in a path
  sims: number of paths
  vol_mult: multiplier that affects volatilty to inject stress
  df: dataframe of the assets adjusted closes
  rand: input the string 'yes' to return a random path, otherwise ignore
  """
  try:
    if T <= 2:
      raise ValueError("The length of each simulated path is too short.")
    elif T < 21:
      print("Warning: Limited price data may lead to unreliable metrics.")
      
    #Intialize dictionary to store simulated paths of T days for each ticker
    tickers = list(df.columns)
    sims_prices = {ticker: np.full(shape=(T, sims), fill_value=0.0) for ticker in tickers}
    
    method = method.strip().lower()

    #Gather inital metrics from historical data
    last_prices = np.array(df.iloc[-1])
    log_returns = np.log(df/df.shift()).dropna()
    cov_matrix = log_returns.cov()*(vol_mult**2)
    L = np.linalg.cholesky(cov_matrix)

    expected_return = log_returns.mean()
    meanM = np.full(shape=(T, len(tickers)), fill_value=expected_return)

    #Generate paths
    for m in range(sims):
      Z = np.random.normal(size=(T, len(tickers)))
      dailyReturns = meanM + Z @ L.T
      cumReturns = (1+dailyReturns).cumprod(axis=0)
      prices = last_prices*cumReturns
      for i, ticker in enumerate(tickers):
        sims_prices[ticker][:,m] = prices[:,i]

    #Get a random path
    if isinstance(rand, str) and rand.strip().lower() == "yes":
      random_int = np.random.randint(0,sims)
      random_sims_prices = {ticker: sims_prices[ticker][:,random_int] for ticker in tickers}
      random_sims_df = pd.DataFrame(random_sims_prices)
      return random_sims_df
    elif rand != None:
      raise ValueError("Invaild input for 'rand'. Input the string 'yes' to return a random path, otherwise ignore.")
    else:
        return sims_prices

  except Exception as e:
    print(f"Error in monte_carlo: {e}")
    return None
  
  
def historical(df:pd.DataFrame, crisis:str):
  """
  Simulates the prices of your portfolio if a historical event were to happen again.

  df: dataframe of the assets adjusted closes
  crisis: string of the event you want to simulate

  Crisis Options:
  "DOT-COM" -- The Dot-Com bubble
  "2008 GFC" -- 2008 Global Financial Crisis
  "2011 Euro" -- 2011 Eurozone Crisis
  "COVID" -- COVID-19 Pandemic
  "2022 Inf" -- 2022 Inflation Crash
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

    last_price = df.iloc[-1]
    crisisReturns = np.log(dfcrisis/dfcrisis.shift()).dropna()
    cumReturns = (1+crisisReturns).cumprod()
    crisisPrices = last_price.mul(cumReturns)
    return crisisPrices

  except Exception as e:
    print(f" \n Error in historical: {e}")
    return None
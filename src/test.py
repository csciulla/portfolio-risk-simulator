from portfolio import Portfolio
from metrics import calculate_metrics
from simulation import monte_carlo
import pandas as pd
import yfinance as yf

test = Portfolio(["TSLA","INTL","GOOG"], 0.0, 0.5)
df = test.get_data()
weights = test.get_weights()

sims_df = monte_carlo(100,100, 2.0, df)
metrics = calculate_metrics(weights, sims_df)
print(metrics)
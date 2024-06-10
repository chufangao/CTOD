Prerequisites:
1. Download the trial dataset from CITI. If it has already been downloaded, provide the path to the data in the scripts.
2. Create a csv file 'tickers.csv' that contains the names and corresponding tickers of sponsors in 'name' and 'ticker' columns respectively.
3. Install yfinance from https://pypi.org/project/yfinance/



Steps:
1. Run 'tickers_2_history.ipynb' to get historical stock prices for the sponsors in 'tickers.csv' if those are available publicly. The stock price data will be stored in 'stock_prices_historical.csv'.
2. Use studies.txt and sponsors.txt from CTTI with stock_prices_historical.csv and tickers.csv to calculate the slope of the stock prices using 'slope_calculation.ipynb'.
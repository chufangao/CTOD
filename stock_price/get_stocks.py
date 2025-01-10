import pandas as pd
from tqdm import tqdm
from time import sleep
import argparse
import os
import yfinance as yf


def get_stocks(TICKERS_PATH, SAVE_STOCKS_PATH):
    ticker_dict = pd.read_csv(TICKERS_PATH)
    ticker_dict

    historical_dataframes = []

    for index, row in tqdm(ticker_dict.iterrows(), total=len(ticker_dict), desc="Processing"):
        company = row['name']
        ticker = row['Ticker']

        stock = yf.Ticker(ticker)

        # Retrieve historical market data
        hist = stock.history(period="max")  # You can adjust the period as needed

        # Add ticker as a new column
        hist['Ticker'] = ticker
        hist['name'] = company
        
        print(f"Retrieved {len(hist)} rows of data for {company} ({ticker})")
        # Save historical data to list
        historical_dataframes.append(hist)
        sleep(1)  # Be nice to Yahoo Finance API

    # Combine all historical dataframes into one
    all_data = pd.concat(historical_dataframes)

    # Reset index to make it cleaner
    all_data.reset_index(inplace=True)
    all_data.head()

    all_data.to_csv(SAVE_STOCKS_PATH, index=False, compression='zip')

def calculate_sma_slope(df, ticker, given_date, window_size, days):
    """
    Calculate the slope of Simple Moving Average (SMA) within a specified number of days from a given date for a given ticker.

    Parameters:
    - df: DataFrame containing historical stock prices with 'Date', 'Ticker', and 'Close' columns.
    - ticker: Ticker symbol of the stock.
    - given_date: Date for which to calculate the slope of SMA within the specified window.
    - window_size: Number of days for the SMA window.
    - days: number of days after the trial completion date

    Returns:
    - slope: Slope of SMA within the specified window from the given date for the given ticker.
    """
    # Convert given_date to a Timestamp object if it's in string format
    if isinstance(given_date, str):
        given_date = pd.to_datetime(given_date, utc=True)

    # Filter DataFrame for the given ticker
    df_ticker = df[df['name'] == ticker]

    if not isinstance(df_ticker['Date'].iloc[0], pd.Timestamp):
        df_ticker['Date'] = pd.to_datetime(df_ticker['Date'], utc=True)

    df_ticker = df_ticker.sort_values(by='Date')

    selected_dates = df_ticker['Close'][df_ticker['Date'] > given_date]

    # Calculate SMA using rolling method only for dates after the given date
    sma = selected_dates.rolling(window=window_size, min_periods = 1).mean()

    # Calculate the slope of SMA within the specified window after the given date
    if len(sma) < days:
        return pd.NA
    slope = (sma.iloc[days-1] - sma.iloc[0]) / window_size

    return slope
    
def process_stocks(CTTI_PATH, SAVE_STOCKS_PATH, SAVE_STOCKS_SLOPES_PATH):
    sponsors = pd.read_csv(os.path.join(CTTI_PATH, 'sponsors.txt'), sep='|')
    studies = pd.read_csv(os.path.join(CTTI_PATH, 'studies.txt'), sep='|')
    stocks_df = pd.read_csv(SAVE_STOCKS_PATH)

    stocks_df['Date'] = pd.to_datetime(stocks_df['Date'], utc=True)

    sponsors['name'] = sponsors['name'].str.lower()
    stocks_df['name'] = stocks_df['name'].str.lower()

    sponsors = sponsors[sponsors['name'].isin(set(stocks_df['name']))]
    sponsors = pd.merge(sponsors, studies[['nct_id', 'completion_date']], on='nct_id', how='inner')
    sponsors['completion_date'] = pd.to_datetime(sponsors['completion_date'], utc=True)

    sponsor_stock_dict = dict(list(stocks_df.groupby('name')))
    tqdm.pandas()
    sponsors['Slope'] = sponsors.progress_apply(lambda row: calculate_sma_slope(sponsor_stock_dict[row['name']], row['name'], row['completion_date'], 5, 7), axis=1)
    sponsors.dropna(subset=['Slope'], inplace=True)

    sponsors.to_csv(SAVE_STOCKS_SLOPES_PATH, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--CTTI_PATH', type=str, default='../CTTI/')
    parser.add_argument('--TICKERS_PATH', type=str, default='./tickers.csv')
    parser.add_argument('--SAVE_STOCKS_PATH', type=str, default='./stock_data.csv.zip')
    parser.add_argument('--SAVE_STOCKS_SLOPES_PATH', type=str, default='./stock_labels.csv')
    args = parser.parse_args()

    get_stocks(args.TICKERS_PATH, args.SAVE_STOCKS_PATH)
    process_stocks(args.CTTI_PATH, args.SAVE_STOCKS_PATH, args.SAVE_STOCKS_SLOPES_PATH)
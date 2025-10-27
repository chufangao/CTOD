# 📈 Stock Price Analysis for Clinical Trial Outcomes

## Overview

This module explores stock price fluctuations as an indicator of market sentiment towards clinical trial outcomes. We hypothesize that stock prices of pharmaceutical and biotech companies often reflect market expectations about trial success or failure.

## 🔧 Prerequisites

### Required Data Sources

1. **CTTI Clinical Trial Data**: Download from [CTTI website](https://aact.ctti-clinicaltrials.org/download)
2. **Company Ticker Data**: Create `tickers.csv` with sponsor names and stock tickers
3. **Yahoo Finance Access**: Via `yfinance` package

### Dependencies

```bash
pip install yfinance pandas numpy matplotlib seaborn jupyter
```

### Required Files

Create `tickers.csv` with the following structure:
```csv
name,ticker
Pfizer Inc,PFE
Johnson & Johnson,JNJ
Merck & Co,MRK
```

## 🚀 Quick Start

### Step 1: Collect Historical Stock Data

```bash
cd stock_price

# Run the stock data collection notebook
jupyter notebook tickers_2_history.ipynb
```

This generates `stock_prices_historical.csv` with historical price data for all available tickers.

### Step 2: Calculate Stock Price Slopes

```bash
# Analyze stock trends around trial completion dates
jupyter notebook slope_calculation.ipynb
```

## 📊 Methodology

### Data Collection Strategy

1. **Company Identification**: Extract sponsors from CTTI studies
2. **Ticker Mapping**: Match company names to stock tickers
3. **Historical Data**: Gather stock prices around trial completion dates
4. **Trend Analysis**: Calculate moving averages and slopes

### Technical Approach

#### Moving Average Calculation
- **Window Size**: 5-day Simple Moving Average (SMA)
- **Purpose**: Reduce short-term price noise
- **Formula**: `SMA = (P₁ + P₂ + P₃ + P₄ + P₅) / 5`

#### Slope Calculation
- **Window**: 7-day period post-trial completion
- **Method**: Linear regression on SMA values
- **Interpretation**: Positive slope = bullish sentiment, Negative = bearish

### Signal Generation

```python
# Pseudo-code for slope calculation
def calculate_stock_signal(stock_data, completion_date):
    # Get 7-day window after completion
    window_data = stock_data[completion_date:completion_date+7]
    
    # Calculate 5-day SMA
    sma = window_data.rolling(window=5).mean()
    
    # Calculate slope using linear regression
    slope = linear_regression(sma).slope
    
    return 1 if slope > threshold else 0
```

## 🔄 Pipeline Steps

### Step 1: Data Preparation

```python
# 1. Load trial completion data
studies = pd.read_csv('studies.txt', sep='|')
sponsors = pd.read_csv('sponsors.txt', sep='|')

# 2. Load ticker mapping
tickers = pd.read_csv('tickers.csv')

# 3. Merge data
trial_sponsor_tickers = merge_trial_sponsor_ticker_data(studies, sponsors, tickers)
```

### Step 2: Stock Data Collection

```python
import yfinance as yf

# Collect historical data for each ticker
for ticker in unique_tickers:
    stock_data = yf.download(ticker, start='2010-01-01', end='2024-01-01')
    save_stock_data(stock_data, ticker)
```

### Step 3: Trend Analysis

```python
# For each trial completion date
for trial in trials:
    completion_date = trial.completion_date
    ticker = trial.sponsor_ticker
    
    # Calculate stock trend signal
    signal = calculate_slope_signal(ticker, completion_date)
    trial_signals.append({'nct_id': trial.nct_id, 'stock_signal': signal})
```

## 📁 File Structure

```
stock_price/
├── README.md                    # This file
├── tickers.csv                  # Company name to ticker mapping
├── tickers_2_history.ipynb      # Stock data collection notebook
├── slope_calculation.ipynb      # Trend analysis notebook
├── get_stocks.py               # Automated stock data retrieval
├── scrape_amendments.py        # Trial amendment analysis
├── stock_prices_historical.csv  # Historical stock data (generated)
└── trial_stock_signals.csv     # Final signals (generated)
```

## ⚙️ Configuration Parameters

### Slope Calculation Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `sma_window` | Moving average window | 5 days | 3-10 days |
| `slope_window` | Trend analysis period | 7 days | 5-14 days |
| `significance_threshold` | Minimum slope magnitude | 0.01 | 0.001-0.1 |
| `volume_threshold` | Minimum trading volume | 1000 | 100-10000 |

### Data Quality Filters

```python
# Example filtering criteria
filters = {
    'min_trading_days': 5,      # Minimum days with trading data
    'max_price_gap': 0.5,       # Maximum allowed price gap (50%)
    'min_volume': 1000,         # Minimum daily volume
    'exclude_splits': True,     # Exclude stock split dates
    'exclude_dividends': True   # Exclude dividend dates
}
```

## 📈 Performance Metrics

### Coverage Statistics

- **Public Companies**: ~200-300 sponsors with public tickers
- **Trial Coverage**: ~30-40% of industry-sponsored trials
- **Data Completeness**: 85-95% for available tickers
- **Signal Quality**: Correlation with trial outcomes reported in paper

### Processing Time

| Task | Dataset Size | Estimated Time |
|------|-------------|----------------|
| Ticker Collection | 1000 sponsors | 2-4 hours |
| Historical Data | 300 tickers | 4-8 hours |
| Slope Calculation | Full dataset | 1-2 hours |

## 🧪 Analysis Examples

### Basic Stock Signal Generation

```python
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def generate_stock_signals(trials_df, tickers_df):
    """Generate stock-based weak labels for trials."""
    signals = []
    
    for _, trial in trials_df.iterrows():
        ticker = get_ticker_for_sponsor(trial.sponsor, tickers_df)
        if ticker:
            signal = calculate_completion_signal(ticker, trial.completion_date)
            signals.append({
                'nct_id': trial.nct_id,
                'stock_signal': signal,
                'confidence': calculate_confidence(ticker, trial.completion_date)
            })
    
    return pd.DataFrame(signals)
```

### Advanced Analysis

```python
# Analyze multiple time windows
def multi_window_analysis(ticker, completion_date):
    """Analyze stock trends across multiple time windows."""
    windows = [7, 14, 30]  # days
    signals = {}
    
    for window in windows:
        slope = calculate_slope(ticker, completion_date, window)
        signals[f'slope_{window}d'] = slope
        
    return signals
```

## 🔗 Integration with Other Modules

### Input Data Sources
- **[Clinical Trial Linkage](../clinical_trial_linkage/)**: Trial completion dates
- **CTTI Data**: Sponsor information and trial metadata

### Output Usage
- **[Labeling Module](../labeling/)**: Stock signals as weak supervision
- **[Baselines](../baselines/)**: Features for baseline models

### Data Format for Integration

```python
# Expected output format
stock_labels = pd.DataFrame({
    'nct_id': ['NCT12345', 'NCT67890'],
    'stock_signal': [1, 0],  # 1: Positive trend, 0: Negative trend
    'slope_magnitude': [0.05, -0.03],
    'confidence': [0.8, 0.6],
    'trading_volume': [15000, 8000]
})
```

## 🐛 Troubleshooting

### Common Issues

**1. Missing Ticker Data**
```python
# Check ticker availability
import yfinance as yf
ticker_info = yf.Ticker("AAPL").info
print("Ticker valid:", ticker_info is not None)
```

**2. Insufficient Historical Data**
```python
# Verify data availability around completion date
stock_data = yf.download("PFE", start=completion_date-timedelta(30), 
                        end=completion_date+timedelta(30))
print(f"Available data points: {len(stock_data)}")
```

**3. Stock Split/Dividend Issues**
```python
# Account for corporate actions
stock_data = yf.download("TICKER", auto_adjust=True)  # Automatically adjust for splits
```

### Data Quality Checks

```python
def validate_stock_data(stock_df):
    """Validate stock data quality."""
    checks = {
        'no_missing_dates': stock_df.isnull().sum().sum() == 0,
        'reasonable_prices': (stock_df['Close'] > 0).all(),
        'sufficient_volume': (stock_df['Volume'] > 100).mean() > 0.8,
        'no_extreme_gaps': (stock_df['Close'].pct_change().abs() < 0.5).mean() > 0.95
    }
    return checks
```

## 📚 Additional Resources

- **yfinance Documentation**: [PyPI](https://pypi.org/project/yfinance/)
- **Financial Data Analysis**: [Quantitative Finance Resources](https://quantlib.org/)
- **Market Data APIs**: [Alternative data sources](https://www.alphavantage.co/)
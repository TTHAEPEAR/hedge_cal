# Hedge Calculator

A Python tool for calculating optimal hedge ratios and evaluating hedge effectiveness using OLS regression and variance reduction methods.

## Features

- **Hedge Ratio Estimation**: Calculate optimal hedge ratios using OLS (Ordinary Least Squares) or Closed-Form (Covariance/Variance) methods.
- **Effectiveness Analysis**: Evaluate the performance of your hedge with In-Sample and Out-of-Sample effectiveness metrics (Variance Reduction).
- **Backtesting**: Simulate hedging strategies over historical data with customizable split ratios.
- **Flexible Returns**: Support for both arithmetic (`diff`) and logarithmic (`log`) return calculations.
- **Utility Functions**: Includes tools for Annualized Sharpe Ratio calculation and Contract Sizing.

## Requirements

- python >= 3.7
- pandas
- numpy
- scikit-learn
- statsmodels

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas numpy scikit-learn statsmodels
   ```

## Usage

### Basic Backtest

Run a backtest on your spot and futures data to estimate the optimal hedge ratio and check effectiveness.

```python
import pandas as pd
from src.hedge_calculator import backtest

# Load your data (must have 'date', 'spot', 'fut' columns)
df = pd.read_csv("data/sample_spot_fut.csv")

# Run backtest
# split=0.7 means 70% of data is used for training (estimation), 30% for testing
res = backtest(df, split=0.7, mode='diff', method='ols')

print(f"Optimal Hedge Ratio: {res.hedge_ratio:.4f}")
print(f"In-Sample Effectiveness: {res.in_sample_effectiveness:.2%}")
if res.out_sample_effectiveness:
    print(f"Out-of-Sample Effectiveness: {res.out_sample_effectiveness:.2%}")
```

### Manual Calculation

You can also use the core functions step-by-step.

```python
import pandas as pd
from src.hedge_calculator import cal_changes, est_h_ratio, apply_h

# 1. Prepare data (calculate price changes)
df = pd.read_csv("data/sample_spot_fut.csv")
df_changes = cal_changes(df, mode='diff')

# 2. Estimate Hedge Ratio
h_ratio, details = est_h_ratio(df_changes, method='ols')
print(f"Hedge Ratio: {h_ratio}")

# 3. Apply Hedge (Calculate Hedged Portfolio Returns)
hedged_returns = apply_h(df_changes, h_ratio)
```

### Contract Sizing

Calculate the number of contracts needed to hedge a specific portfolio exposure.

```python
from src.hedge_calculator import contract_size

exposure = 1_000_000  # Value of portfolio to hedge
fut_price = 100.0     # Current futures price
multiplier = 1000     # Contract multiplier
h_ratio = 0.95        # Estimated hedge ratio

num_contracts = contract_size(exposure, fut_price, multiplier, h_ratio)
print(f"Contracts needed: {num_contracts}")
```

## Project Structure

- `src/`: Contains the core logic in `hedge_calculator.py`.
- `data/`: Sample data for testing and demonstration.
- `tests/`: Basic test scripts to verify functionality.

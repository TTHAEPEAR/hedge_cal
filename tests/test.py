import pandas as pd
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.hedge_calculator import backtest, cal_changes, est_h_ratio, apply_h, ann_sharpe

df = pd.read_csv("data/sample_spot_fut.csv")
  # has columns: date, spot, fut
res = backtest(df, split=0.7, mode='diff', method='ols')

print("Hedge ratio (h*):", res.hedge_ratio)
print("Method:", res.method)
print("In-sample effectiveness:", res.in_sample_effectiveness)
print("Out-of-sample effectiveness:", res.out_sample_effectiveness)
# Optional: OLS diagnostics
if "r2" in res.details:
    print("OLS R^2:", res.details["r2"])

# Build hedged series and a Sharpe
chg = cal_changes(df, mode='diff')
hedged = apply_h(chg, res.hedge_ratio)
print("Annualized Sharpe (hedged Δ):", ann_sharpe(hedged))

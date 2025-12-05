
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from sklearn.metrics import r2_score
import statsmodels.api as sm

@dataclass
class HedgeResult:
    hedge_ratio: float
    method: str
    in_sample_effectiveness: float
    out_sample_effectiveness: Optional[float]
    details: dict

def  ensure_date_index(df :pd.DataFrame) -> pd.DataFrame:      
    if 'date' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').set_index('date')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Provide a 'date' column or set a DatetimeIndex.")
    return df
def cal_changes(df : pd.DataFrame, mode: str = 'diff') -> pd.DataFrame:
    #arithmetic differences
    df = ensure_date_index(df)
    if not {'spot','fut'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'spot' and 'fut' columns.")
    sf_data = df[['spot','fut']].copy()
    if mode == 'diff':
        sf_data['dS'] = sf_data['spot'].diff()
        sf_data['dF'] = sf_data['fut'].diff()
    elif mode == 'log':
        sf_data['dS'] = np.log(sf_data['spot']).diff()
        sf_data['dF'] = np.log(sf_data['fut']).diff()
    else :
        raise ValueError("mode must be 'diff' or 'log'")
    return sf_data.dropna()
def est_h_ratio(df = pd.DataFrame, method: str = 'ols', intercept : bool = True) -> Tuple[float,Dict]:
    if not {'dS','dF'}.issubset(df.columns):
        raise ValueError("provide columns with 'dS' and 'dF'")
    if method == 'ols':
        x = df[['dF']].copy()
        if intercept:
            x = sm.add_constant(x)
        model = sm.OLS(df['dS'], x).fit()
        h = model.params['dF']
        r2 = model.rsquared
        return  float(h), {'ols_summary': str(model.summary()), 'r2': float(r2)}
    elif method == 'ClosedForm':
        cov =  np.cov(df['dS'], df['dF'],ddof=1)[1,0]
        varF = np.var(df['dF'], ddof=1)
        h = cov / varF
        return float(h), {'cov': cov, 'varF': varF}
    else :
        raise ValueError("method must be 'ClosedForm' or 'ols'")

def apply_h(df:pd.Series, h_ratio : float) -> pd.Series:
    return df['dS'] - (h_ratio*df['dF'])
def effectiveness(hedge:pd.DataFrame, unhedge : pd.DataFrame) -> float:
    var_u = np.var(unhedge,ddof=1)
    var_h = np.var(hedge,ddof=1)
    if var_u == 0:
        return np.nan
    return 1.0 - (var_h/var_u)
def backtest(df:pd.DataFrame, mode ='diff', method = 'ols',split : float = 0.7) -> HedgeResult:
    df = cal_changes(df, mode=mode)
    n = len(df)
    n_train = int(split*n)
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]
    h , detail = est_h_ratio(train, method)
    hedged_train = apply_h(train, h)
    eff_sample = effectiveness(hedged_train, train['dS'])
    eff_out_sample = None
    if len(test) > 5:
        hedged_test = apply_h(test, h)
        eff_out_sample = effectiveness(hedged_test, test['dS'])
    return HedgeResult(h, method, eff_sample, eff_out_sample, detail)
def ann_sharpe(df :pd.Series, day_per_year :int =252) -> float :
    ann = df.mean()*(day_per_year)
    sd = df.std(ddof=1)*(np.sqrt(day_per_year))
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(ann/sd)
def contract_size(exposure: float ,fut_price: float , multiplier : float ,h_ratio: float , round_to : int =1) -> int:
    if fut_price <= 0 or multiplier <= 0:
        raise ValueError("fut_price and multiplier must be positive.")
    raw_contracts = (exposure * h_ratio) / (multiplier * fut_price)
    if round_to <= 0:
        return int(round(raw_contracts))
    return  int(round(raw_contracts /round_to)*round_to)

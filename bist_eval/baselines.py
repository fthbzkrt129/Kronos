"""Deterministic close-price forecasting baselines."""
from __future__ import annotations
import numpy as np, pandas as pd
BASELINE_METHODS=("last_close","momentum_20","linear_trend_20")
def forecast_baselines(context: pd.DataFrame, horizon: int):
    if horizon<=0: raise ValueError("horizon must be positive")
    close=np.asarray(context["close"],dtype=float)
    if len(close)<20: raise ValueError("at least 20 close rows are required")
    if not np.isfinite(close).all(): raise ValueError("close contains non-finite values")
    last=np.repeat(close[-1],horizon)
    if close[-20]<=0 or close[-1]<=0: raise ValueError("momentum endpoints must be positive")
    rate=(close[-1]/close[-20])**(1.0/19.0)-1.0
    momentum=np.array([close[-1]*(1.0+rate)**h for h in range(1,horizon+1)])
    slope,intercept=np.polyfit(np.arange(20,dtype=float),close[-20:],1)
    trend=slope*np.arange(20,20+horizon,dtype=float)+intercept
    return {"last_close":last,"momentum_20":momentum,"linear_trend_20":trend}

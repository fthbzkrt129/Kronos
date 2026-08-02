"""Leakage-free forecast-window construction."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, pandas as pd
from .calendar import MonthlyCohort
VALUE_COLUMNS=["open","high","low","close","volume","amount"]
@dataclass(frozen=True, slots=True)
class ForecastWindow:
    symbol: str
    candidate_month: str
    forecast_origin: pd.Timestamp
    target_timestamps: tuple[pd.Timestamp,...]
    context: pd.DataFrame
    target: pd.DataFrame
@dataclass(frozen=True, slots=True)
class SkipRecord:
    symbol: str
    candidate_month: str
    reason_code: str
    reason_detail: str
    available_history_rows: int
    available_target_rows: int
def build_symbol_windows(symbol: str, frame: pd.DataFrame, cohorts: list[MonthlyCohort], *, lookback: int, horizon: int):
    indexed=frame.set_index("timestamps",drop=False); windows=[]; skips=[]
    for cohort in cohorts:
        if cohort.forecast_origin not in indexed.index:
            skips.append(SkipRecord(symbol,cohort.candidate_month,"missing_origin_date","symbol does not contain common forecast origin",0,0)); continue
        missing=[t for t in cohort.target_timestamps if t not in indexed.index]
        if missing:
            skips.append(SkipRecord(symbol,cohort.candidate_month,"missing_target_date",f"missing {len(missing)} common target date(s)",int((frame.timestamps<=cohort.forecast_origin).sum()),horizon-len(missing))); continue
        origin_pos=indexed.index.get_loc(cohort.forecast_origin)
        if not isinstance(origin_pos,(int,np.integer)): raise ValueError("duplicate origin timestamp")
        history_count=origin_pos+1
        if history_count<lookback:
            skips.append(SkipRecord(symbol,cohort.candidate_month,"insufficient_history",f"need {lookback} rows",history_count,horizon)); continue
        context=frame.iloc[history_count-lookback:history_count].copy().reset_index(drop=True)
        target=indexed.loc[list(cohort.target_timestamps)].copy().reset_index(drop=True)
        if not np.isfinite(context[VALUE_COLUMNS].to_numpy(dtype=float)).all():
            skips.append(SkipRecord(symbol,cohort.candidate_month,"invalid_input_values","context contains non-finite values",history_count,horizon)); continue
        if not np.isfinite(target[VALUE_COLUMNS].to_numpy(dtype=float)).all():
            skips.append(SkipRecord(symbol,cohort.candidate_month,"invalid_target_values","target contains non-finite values",history_count,horizon)); continue
        windows.append(ForecastWindow(symbol,cohort.candidate_month,cohort.forecast_origin,cohort.target_timestamps,context,target))
    return windows,skips

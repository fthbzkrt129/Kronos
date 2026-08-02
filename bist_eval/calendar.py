"""Timestamp-only canonical market calendar and common monthly cohorts."""
from __future__ import annotations
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
import pandas as pd
@dataclass(frozen=True, slots=True)
class MonthlyCohort:
    candidate_month: str
    forecast_origin: pd.Timestamp
    target_timestamps: tuple[pd.Timestamp,...]
@dataclass(frozen=True, slots=True)
class CalendarSkip:
    candidate_month: str
    reason_code: str
    reason_detail: str
def build_canonical_calendar(timestamp_coverage: Mapping[str,pd.DatetimeIndex], *, coverage_threshold: float, start_date: str, end_date: str) -> pd.DatetimeIndex:
    if not timestamp_coverage: return pd.DatetimeIndex([])
    if not 0 < coverage_threshold <= 1: raise ValueError("coverage_threshold must be in (0, 1]")
    counter=Counter()
    for _,index in sorted(timestamp_coverage.items()):
        for ts in pd.DatetimeIndex(index).normalize().unique(): counter[pd.Timestamp(ts)] += 1
    required=ceil(len(timestamp_coverage)*coverage_threshold)
    start=pd.Timestamp(start_date); end=pd.Timestamp(end_date)
    dates=sorted(ts for ts,count in counter.items() if count>=required and start<=ts<=end)
    return pd.DatetimeIndex(dates)
def build_monthly_cohorts(calendar: pd.DatetimeIndex, *, horizon: int):
    if horizon<=0: raise ValueError("horizon must be positive")
    cal=pd.DatetimeIndex(calendar).sort_values().unique(); cohorts=[]; skips=[]
    if len(cal)==0: return cohorts,skips
    periods=pd.Series(cal).dt.to_period("M")
    for period in periods.drop_duplicates():
        positions=[i for i,p in enumerate(periods) if p==period]
        origin_idx=positions[0]; origin=pd.Timestamp(cal[origin_idx]); targets=tuple(pd.Timestamp(x) for x in cal[origin_idx+1:origin_idx+1+horizon])
        if len(targets)!=horizon:
            skips.append(CalendarSkip(str(period),"incomplete_target_calendar",f"need {horizon} target dates, found {len(targets)}")); continue
        cohorts.append(MonthlyCohort(str(period),origin,targets))
    return cohorts,skips

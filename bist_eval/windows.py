from __future__ import annotations
from dataclasses import dataclass
import numpy as np,pandas as pd
from .calendar import MonthlyCohort
VALUE_COLUMNS=["open","high","low","close","volume","amount"]
@dataclass(frozen=True,slots=True)
class ForecastWindow:
    symbol:str; candidate_month:str; forecast_origin:pd.Timestamp; target_timestamps:tuple[pd.Timestamp,...]; context:pd.DataFrame; target:pd.DataFrame
@dataclass(frozen=True,slots=True)
class PredictionWindow:
    symbol:str; candidate_month:str; forecast_origin:pd.Timestamp; target_timestamps:tuple[pd.Timestamp,...]; context:pd.DataFrame
@dataclass(frozen=True,slots=True)
class ScoringRecord:
    symbol:str; candidate_month:str; forecast_origin:pd.Timestamp; target_timestamps:tuple[pd.Timestamp,...]; raw_target:pd.DataFrame; target_provider_factors:np.ndarray
@dataclass(frozen=True,slots=True)
class BenchmarkWindowBundle:
    symbol:str; candidate_month:str; forecast_origin:pd.Timestamp; target_timestamps:tuple[pd.Timestamp,...]; raw_context:pd.DataFrame; context_provider_factors:np.ndarray; scoring_record:ScoringRecord
    def prediction_window(self,context): return PredictionWindow(self.symbol,self.candidate_month,self.forecast_origin,self.target_timestamps,context)
@dataclass(frozen=True,slots=True)
class SkipRecord:
    symbol:str; candidate_month:str; reason_code:str; reason_detail:str; available_history_rows:int; available_target_rows:int
def _positions(frame,cohort):
    indexed=frame.set_index("timestamps",drop=False)
    if cohort.forecast_origin not in indexed.index:return indexed,None,None,"missing_origin_date"
    missing=[t for t in cohort.target_timestamps if t not in indexed.index]
    if missing:return indexed,None,missing,"missing_target_date"
    pos=indexed.index.get_loc(cohort.forecast_origin)
    if not isinstance(pos,(int,np.integer)): raise ValueError("duplicate origin timestamp")
    return indexed,int(pos),None,None
def build_symbol_windows(symbol,frame,cohorts,*,lookback,horizon):
    windows=[];skips=[]
    for c in cohorts:
        indexed,pos,missing,reason=_positions(frame,c)
        if reason=="missing_origin_date":skips.append(SkipRecord(symbol,c.candidate_month,reason,"symbol does not contain common forecast origin",0,0));continue
        if reason=="missing_target_date":skips.append(SkipRecord(symbol,c.candidate_month,reason,f"missing {len(missing)} common target date(s)",int((frame.timestamps<=c.forecast_origin).sum()),horizon-len(missing)));continue
        hc=pos+1
        if hc<lookback:skips.append(SkipRecord(symbol,c.candidate_month,"insufficient_history",f"need {lookback} rows",hc,horizon));continue
        context=frame.iloc[hc-lookback:hc].copy().reset_index(drop=True);target=indexed.loc[list(c.target_timestamps)].copy().reset_index(drop=True)
        if not np.isfinite(context[VALUE_COLUMNS].to_numpy(float)).all():skips.append(SkipRecord(symbol,c.candidate_month,"invalid_input_values","context contains non-finite values",hc,horizon));continue
        if not np.isfinite(target[VALUE_COLUMNS].to_numpy(float)).all():skips.append(SkipRecord(symbol,c.candidate_month,"invalid_target_values","target contains non-finite values",hc,horizon));continue
        windows.append(ForecastWindow(symbol,c.candidate_month,c.forecast_origin,c.target_timestamps,context,target))
    return windows,skips
def build_benchmark_windows(symbol,raw_frame,cohorts,*,lookback,horizon):
    from .adjustments import validate_provider_factors
    factors=validate_provider_factors(raw_frame); bundles=[];skips=[]; indexed=raw_frame.set_index("timestamps",drop=False)
    for c in cohorts:
        _,pos,missing,reason=_positions(raw_frame,c)
        if reason=="missing_origin_date":skips.append(SkipRecord(symbol,c.candidate_month,reason,"symbol does not contain common forecast origin",0,0));continue
        if reason=="missing_target_date":skips.append(SkipRecord(symbol,c.candidate_month,reason,f"missing {len(missing)} common target date(s)",int((raw_frame.timestamps<=c.forecast_origin).sum()),horizon-len(missing)));continue
        hc=pos+1
        if hc<lookback:skips.append(SkipRecord(symbol,c.candidate_month,"insufficient_history",f"need {lookback} rows",hc,horizon));continue
        raw_context=raw_frame.iloc[hc-lookback:hc].copy().reset_index(drop=True); cf=factors.iloc[hc-lookback:hc].to_numpy(float)
        raw_target=indexed.loc[list(c.target_timestamps)].copy().reset_index(drop=True); tf=(raw_target.adj_close/raw_target.close).to_numpy(float)
        score=ScoringRecord(symbol,c.candidate_month,c.forecast_origin,c.target_timestamps,raw_target,tf)
        bundles.append(BenchmarkWindowBundle(symbol,c.candidate_month,c.forecast_origin,c.target_timestamps,raw_context,cf,score))
    return bundles,skips

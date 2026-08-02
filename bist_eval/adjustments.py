from __future__ import annotations
from dataclasses import asdict,dataclass
import hashlib,struct
import numpy as np,pandas as pd
from bist_data.quality import repair_ohlc_envelope,validate_candles
PRICE_COLUMNS=["open","high","low","close"]
@dataclass(frozen=True,slots=True)
class FactorDiagnostics:
    symbol:str; rows:int; first_timestamp:pd.Timestamp; last_timestamp:pd.Timestamp; minimum_factor:float; maximum_factor:float; materially_changed_rows:int; factor_fingerprint:str
@dataclass(frozen=True,slots=True)
class ExposureDiagnostics:
    context_factor_changed:bool; target_factor_changed:bool; context_max_abs_log_step:float; target_max_abs_log_from_origin:float; exposure_bucket:str
def provider_factor(frame): return pd.to_numeric(frame.adj_close,errors="coerce")/pd.to_numeric(frame.close,errors="coerce")
def _factor_fingerprint(timestamps,factors):
    d=hashlib.sha256()
    for ts,v in zip(pd.DatetimeIndex(timestamps),np.asarray(factors,dtype=np.float64)):
        d.update(pd.Timestamp(ts).isoformat().encode());d.update(struct.pack(">d",float(v)))
    return d.hexdigest()
def validate_provider_factors(frame):
    f=provider_factor(frame)
    if not np.isfinite(f.to_numpy(float)).all() or (f<=0).any(): raise ValueError("provider factors must be finite and strictly positive")
    return pd.Series(f.to_numpy(float),index=frame.index,name="provider_factor")
def build_factor_diagnostics(symbol,frame,tolerance):
    if tolerance<=0: raise ValueError("tolerance must be positive")
    f=validate_provider_factors(frame)
    return FactorDiagnostics(symbol,len(frame),pd.Timestamp(frame.timestamps.iloc[0]),pd.Timestamp(frame.timestamps.iloc[-1]),float(f.min()),float(f.max()),int((np.abs(f-1)>tolerance).sum()),_factor_fingerprint(frame.timestamps,f))
def aggregate_factor_fingerprint(records):
    d=hashlib.sha256()
    for r in sorted(records,key=lambda x:x.symbol): d.update(r.symbol.encode());d.update(r.factor_fingerprint.encode())
    return d.hexdigest()
def rebase_context(raw_context,factors,origin_factor):
    factors=np.asarray(factors,dtype=float)
    if len(raw_context)!=len(factors): raise ValueError("context and factor lengths differ")
    if not np.isfinite(origin_factor) or origin_factor<=0: raise ValueError("origin_factor must be positive and finite")
    rel=factors/origin_factor; out=raw_context.loc[:,["timestamps","open","high","low","close","volume"]].copy()
    for c in PRICE_COLUMNS: out[c]=pd.to_numeric(out[c],errors="raise").to_numpy(float)*rel
    out["amount"]=((out.high+out.low+out.close)/3)*pd.to_numeric(out.volume,errors="raise")
    repaired,repairs=repair_ohlc_envelope(out); validated=validate_candles(repaired)
    if (validated[PRICE_COLUMNS]<=0).any().any(): raise ValueError("rebased prices must be positive")
    amount_by_ts=repaired.set_index("timestamps").amount; validated["amount"]=validated.timestamps.map(amount_by_ts)
    return validated.loc[:,["timestamps","open","high","low","close","volume","amount"]],repairs
def transform_target_after_prediction(raw_target,target_factors,origin_factor):
    target_factors=np.asarray(target_factors,dtype=float)
    if len(raw_target)!=len(target_factors): raise ValueError("target and factor lengths differ")
    rel=target_factors/origin_factor; out=raw_target.loc[:,["timestamps","open","high","low","close","volume"]].copy()
    for c in PRICE_COLUMNS: out[c]=pd.to_numeric(out[c],errors="raise").to_numpy(float)*rel
    out["amount"]=((out.high+out.low+out.close)/3)*pd.to_numeric(out.volume,errors="raise")
    repaired,_=repair_ohlc_envelope(out); validated=validate_candles(repaired); amount=repaired.set_index("timestamps").amount; validated["amount"]=validated.timestamps.map(amount)
    return validated.loc[:,["timestamps","open","high","low","close","volume","amount"]]
def classify_exposure(context_factors,target_factors,origin_factor,tolerance):
    cf=np.asarray(context_factors,dtype=float); tf=np.asarray(target_factors,dtype=float)
    cmax=float(np.max(np.abs(np.diff(np.log(cf))))) if len(cf)>1 else 0.; tmax=float(np.max(np.abs(np.log(tf/origin_factor)))) if len(tf) else 0.
    cc=bool(cmax>tolerance); tc=bool(tmax>tolerance); bucket="material_factor_change" if cc or tc else "no_material_change"
    return ExposureDiagnostics(cc,tc,cmax,tmax,bucket)
def factor_manifest_record(d):
    x=asdict(d); x["first_timestamp"]=d.first_timestamp.isoformat();x["last_timestamp"]=d.last_timestamp.isoformat();return x

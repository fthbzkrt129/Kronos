from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import hashlib,json
import numpy as np,pandas as pd
from .adjustments import classify_exposure,rebase_context,transform_target_after_prediction
from .baselines import forecast_baselines
from .metrics import compute_benchmark_window_metrics
from .windows import build_benchmark_windows
from .reporting import BENCHMARK_PREDICTION_COLUMNS,SKIP_COLUMNS
@dataclass(frozen=True,slots=True)
class BenchmarkShardResult:
    predictions:pd.DataFrame;window_metrics:pd.DataFrame;skips:pd.DataFrame;manifest:dict
def _raw_context(raw):
    out=raw.loc[:,["timestamps","open","high","low","close","volume"]].copy();out["amount"]=((out.high+out.low+out.close)/3)*out.volume;return out
def _target_fp(target):
    payload=[(pd.Timestamp(t).isoformat(),float(c)) for t,c in zip(target.timestamps,target.close)]
    return hashlib.sha256(json.dumps(payload,separators=(",",":")).encode()).hexdigest()
def _prediction_rows(arm,method,window,pred,actual,last,exposure,target_fp,context_view):
    rows=[]
    for step,(ts,p,a) in enumerate(zip(window.target_timestamps,pred,actual),1):
        rows.append({"experiment_arm":arm,"symbol":window.symbol,"candidate_month":window.candidate_month,"forecast_origin":window.forecast_origin,"target_timestamp":ts,"horizon_step":step,"method":method,"predicted_close":float(p),"actual_close":float(a),"history_last_close":float(last),"context_view":context_view,"scoring_target_view":"origin_rebased","exposure_bucket":exposure.exposure_bucket,"context_factor_changed":exposure.context_factor_changed,"target_factor_changed":exposure.target_factor_changed,"context_max_abs_log_step":exposure.context_max_abs_log_step,"target_max_abs_log_from_origin":exposure.target_max_abs_log_from_origin,"common_target_fingerprint":target_fp})
    return rows
def _prepare(raw_frames,cohorts,symbols,lookback,horizon,tolerance):
    groups=defaultdict(list);skip_rows=[]
    for symbol in symbols:
        bundles,skips=build_benchmark_windows(symbol,raw_frames[symbol],cohorts,lookback=lookback,horizon=horizon);skip_rows.extend(s.__dict__ for s in skips)
        for b in bundles:
            origin_factor=float(b.context_provider_factors[-1]);rebased,repairs=rebase_context(b.raw_context,b.context_provider_factors,origin_factor);raw=_raw_context(b.raw_context);exposure=classify_exposure(b.context_provider_factors,b.scoring_record.target_provider_factors,origin_factor,tolerance)
            groups[(b.forecast_origin,b.target_timestamps)].append((b,raw,rebased,origin_factor,exposure,repairs))
    return groups,skip_rows
def run_mini_pair_shard(*,raw_frames,cohorts,symbols,config,adapter,manifest_base=None):
    groups,skip_rows=_prepare(raw_frames,cohorts,symbols,config.lookback,config.horizon,config.material_factor_tolerance);rows=[]
    for _,items in sorted(groups.items(),key=lambda kv:kv[0][0]):
        raw_windows=[b.prediction_window(raw) for b,raw,rebased,origin,exp,rep in items];adjusted_windows=[b.prediction_window(rebased) for b,raw,rebased,origin,exp,rep in items]
        raw_out=adapter.predict_cohort(raw_windows);adjusted_out=adapter.predict_cohort(adjusted_windows)
        for b,raw,rebased,origin_factor,exposure,_ in items:
            actual_frame=transform_target_after_prediction(b.scoring_record.raw_target,b.scoring_record.target_provider_factors,origin_factor);actual=actual_frame.close.to_numpy(float);last=float(raw.close.iloc[-1]);fp=_target_fp(actual_frame)
            rows.extend(_prediction_rows("raw-mini","kronos",b,raw_out[b.symbol],actual,last,exposure,fp,"raw"));rows.extend(_prediction_rows("adjusted-mini","kronos",b,adjusted_out[b.symbol],actual,last,exposure,fp,"origin_rebased"))
            for method,pred in forecast_baselines(rebased,config.horizon).items():rows.extend(_prediction_rows("adjusted-baselines",method,b,pred,actual,last,exposure,fp,"origin_rebased"))
    pred=pd.DataFrame(rows,columns=BENCHMARK_PREDICTION_COLUMNS);wm=compute_benchmark_window_metrics(pred) if len(pred) else pd.DataFrame();skips=pd.DataFrame(skip_rows,columns=SKIP_COLUMNS)
    manifest={**(manifest_base or {}),"mode":"mini-pair","experiment_arms":["raw-mini","adjusted-mini","adjusted-baselines"],"symbols":list(symbols),"prediction_rows":len(pred),"eligible_windows":int(((wm.experiment_arm=="raw-mini")&(wm.method=="kronos")).sum()) if len(wm) else 0}
    return BenchmarkShardResult(pred,wm,skips,manifest)
def run_small_shard(*,raw_frames,cohorts,symbols,config,adapter,manifest_base=None):
    groups,skip_rows=_prepare(raw_frames,cohorts,symbols,config.lookback,config.horizon,config.material_factor_tolerance);rows=[]
    for _,items in sorted(groups.items(),key=lambda kv:kv[0][0]):
        windows=[b.prediction_window(rebased) for b,raw,rebased,origin,exp,rep in items];outs=adapter.predict_cohort(windows)
        for b,raw,rebased,origin_factor,exposure,_ in items:
            actual_frame=transform_target_after_prediction(b.scoring_record.raw_target,b.scoring_record.target_provider_factors,origin_factor);actual=actual_frame.close.to_numpy(float);last=float(raw.close.iloc[-1]);fp=_target_fp(actual_frame)
            rows.extend(_prediction_rows("adjusted-small","kronos",b,outs[b.symbol],actual,last,exposure,fp,"origin_rebased"))
    pred=pd.DataFrame(rows,columns=BENCHMARK_PREDICTION_COLUMNS);wm=compute_benchmark_window_metrics(pred) if len(pred) else pd.DataFrame();skips=pd.DataFrame(skip_rows,columns=SKIP_COLUMNS)
    manifest={**(manifest_base or {}),"mode":"small","experiment_arms":["adjusted-small"],"symbols":list(symbols),"prediction_rows":len(pred),"eligible_windows":int((wm.method=="kronos").sum()) if len(wm) else 0}
    return BenchmarkShardResult(pred,wm,skips,manifest)

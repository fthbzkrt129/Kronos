from __future__ import annotations
from dataclasses import dataclass
import numpy as np,pandas as pd
@dataclass(frozen=True,slots=True)
class ComparisonSpec:
    comparison_id:str; challenger_arm:str; challenger_method:str; reference_arm:str; reference_method:str
DEFAULT_COMPARISONS=(
 ComparisonSpec("adjusted-mini_vs_raw-mini","adjusted-mini","kronos","raw-mini","kronos"),
 ComparisonSpec("adjusted-small_vs_adjusted-mini","adjusted-small","kronos","adjusted-mini","kronos"),
 ComparisonSpec("adjusted-mini_vs_last-close","adjusted-mini","kronos","adjusted-baselines","last_close"),
 ComparisonSpec("adjusted-mini_vs_momentum","adjusted-mini","kronos","adjusted-baselines","momentum_20"),
 ComparisonSpec("adjusted-mini_vs_linear-trend","adjusted-mini","kronos","adjusted-baselines","linear_trend_20"),
 ComparisonSpec("adjusted-small_vs_last-close","adjusted-small","kronos","adjusted-baselines","last_close"),
 ComparisonSpec("adjusted-small_vs_momentum","adjusted-small","kronos","adjusted-baselines","momentum_20"),
 ComparisonSpec("adjusted-small_vs_linear-trend","adjusted-small","kronos","adjusted-baselines","linear_trend_20"),
)
def _select(wm,arm,method,prefix):
    cols=["symbol","candidate_month","forecast_origin","log_return_abs_error","final_target_timestamp","history_last_close","actual_final_close","scoring_target_view","exposure_bucket","common_target_fingerprint"]
    x=wm[(wm.experiment_arm==arm)&(wm.method==method)].loc[:,cols].copy()
    return x.rename(columns={"log_return_abs_error":f"{prefix}_error"})
def build_paired_comparison(window_metrics,spec):
    c=_select(window_metrics,spec.challenger_arm,spec.challenger_method,"challenger");r=_select(window_metrics,spec.reference_arm,spec.reference_method,"reference")
    keys=["symbol","candidate_month","forecast_origin"];m=c.merge(r,on=keys,suffixes=("_challenger","_reference"),how="inner",validate="one_to_one")
    checks=["final_target_timestamp","history_last_close","actual_final_close","scoring_target_view","exposure_bucket","common_target_fingerprint"]
    for col in checks:
        a=m[f"{col}_challenger"];b=m[f"{col}_reference"]
        if pd.api.types.is_numeric_dtype(a):ok=np.isclose(a.astype(float),b.astype(float),equal_nan=True)
        else:ok=a.astype(str).to_numpy()==b.astype(str).to_numpy()
        if not np.all(ok):raise ValueError(f"paired target mismatch: {col}")
    m["comparison_id"]=spec.comparison_id;m["challenger_arm"]=spec.challenger_arm;m["challenger_method"]=spec.challenger_method;m["reference_arm"]=spec.reference_arm;m["reference_method"]=spec.reference_method
    m["difference"]=m.challenger_error-m.reference_error;m["winner"]=np.where(m.difference<0,"challenger",np.where(m.difference>0,"reference","tie"))
    m["exposure_bucket"]=m.exposure_bucket_challenger;m["common_target_fingerprint"]=m.common_target_fingerprint_challenger
    return m[["comparison_id","symbol","candidate_month","forecast_origin","challenger_arm","challenger_method","reference_arm","reference_method","challenger_error","reference_error","difference","winner","exposure_bucket","common_target_fingerprint"]]
def clustered_bootstrap_mean_difference(paired,*,cluster_column,draws,confidence,seed):
    if draws<=0 or not 0<confidence<1:raise ValueError("invalid bootstrap parameters")
    if cluster_column not in paired:raise ValueError("missing cluster column")
    if not np.isfinite(paired.difference.to_numpy(float)).all():raise ValueError("non-finite paired difference")
    clusters=pd.Index(paired[cluster_column].drop_duplicates())
    if len(clusters)<2:return {"available":False,"cluster_column":cluster_column,"rows":len(paired),"clusters":len(clusters),"draws":draws,"mean_difference":float(paired.difference.mean()) if len(paired) else np.nan,"lower":np.nan,"upper":np.nan}
    grouped={c:paired.loc[paired[cluster_column]==c,"difference"].to_numpy(float) for c in clusters};rng=np.random.default_rng(seed);means=np.empty(draws)
    for i in range(draws):
        sample=rng.choice(clusters.to_numpy(),size=len(clusters),replace=True);values=np.concatenate([grouped[c] for c in sample]);means[i]=values.mean()
    alpha=(1-confidence)/2;lower,upper=np.quantile(means,[alpha,1-alpha])
    return {"available":True,"cluster_column":cluster_column,"rows":len(paired),"clusters":len(clusters),"draws":draws,"mean_difference":float(paired.difference.mean()),"lower":float(lower),"upper":float(upper)}
def summarize_comparison_intervals(paired,*,draws,confidence,seed):
    symbol=clustered_bootstrap_mean_difference(paired,cluster_column="symbol",draws=draws,confidence=confidence,seed=seed)
    origin=clustered_bootstrap_mean_difference(paired,cluster_column="forecast_origin",draws=draws,confidence=confidence,seed=seed+1)
    if symbol["available"] and origin["available"] and symbol["upper"]<0 and origin["upper"]<0:decision="robustly_better"
    elif symbol["available"] and origin["available"] and symbol["lower"]>0 and origin["lower"]>0:decision="robustly_worse"
    else:decision="mixed_or_inconclusive"
    rows=[]
    for kind,result in (("symbol",symbol),("forecast_origin",origin)):
        rows.append({"comparison_id":paired.comparison_id.iloc[0] if len(paired) else None,"cluster_type":kind,**result,"decision":decision})
    return pd.DataFrame(rows)

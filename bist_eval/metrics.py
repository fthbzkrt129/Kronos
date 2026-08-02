"""Guarded per-window and aggregate evaluation metrics."""
from __future__ import annotations
import numpy as np, pandas as pd
def _safe_corr(a,b,method="pearson"):
    frame=pd.DataFrame({"a":a,"b":b}).replace([np.inf,-np.inf],np.nan).dropna()
    if len(frame)<2 or frame.a.nunique()<2 or frame.b.nunique()<2: return np.nan
    if method == "spearman":
        frame = frame.rank(method="average")
    elif method != "pearson":
        raise ValueError(f"unsupported correlation method: {method}")
    return float(frame.a.corr(frame.b,method="pearson"))
def compute_window_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows=[]; keys=["symbol","candidate_month","forecast_origin","method"]
    for key,g in predictions.sort_values("horizon_step").groupby(keys,dropna=False):
        pred=g.predicted_close.to_numpy(float); actual=g.actual_close.to_numpy(float); last=float(g.history_last_close.iloc[0])
        mae=float(np.mean(np.abs(pred-actual))); rmse=float(np.sqrt(np.mean((pred-actual)**2))); final_abs=float(abs(pred[-1]-actual[-1]))
        ape=np.nan if actual[-1]==0 else float(final_abs/abs(actual[-1])); pr=np.nan if last==0 else float(pred[-1]/last-1); ar=np.nan if last==0 else float(actual[-1]/last-1)
        direction=np.nan if not np.isfinite(pr) or not np.isfinite(ar) else bool(np.sign(pr)==np.sign(ar))
        rows.append(dict(zip(keys,key),mae=mae,rmse=rmse,final_ape=ape,final_abs_error=final_abs,predicted_return_5d=pr,actual_return_5d=ar,direction_correct=direction))
    return pd.DataFrame(rows)
def aggregate_symbol_metrics(wm):
    rows=[]
    for (symbol,method),g in wm.groupby(["symbol","method"]):
        rows.append({"symbol":symbol,"method":method,"window_count":len(g),"mean_mae":g.mae.mean(),"median_mae":g.mae.median(),"mean_rmse":g.rmse.mean(),"mean_final_ape":g.final_ape.mean(),"direction_accuracy":pd.to_numeric(g.direction_correct,errors="coerce").mean(),"return_pearson":_safe_corr(g.predicted_return_5d,g.actual_return_5d)})
    return pd.DataFrame(rows)
def aggregate_period_metrics(wm):
    return wm.groupby(["candidate_month","forecast_origin","method"],as_index=False).agg(symbol_count=("symbol","nunique"),mean_mae=("mae","mean"),mean_rmse=("rmse","mean"),direction_accuracy=("direction_correct","mean"))
def compute_ranking_metrics(wm, minimum_cohort):
    rows=[]
    for (month,origin,method),g in wm.groupby(["candidate_month","forecast_origin","method"]):
        g=g.dropna(subset=["predicted_return_5d","actual_return_5d"]); n=g.symbol.nunique()
        if n<minimum_cohort:
            rows.append({"candidate_month":month,"forecast_origin":origin,"method":method,"eligible_symbols":n,"spearman":np.nan,"top5_overlap":np.nan,"predicted_top5_mean_realized_return":np.nan,"ranking_available":False}); continue
        top_pred=set(g.nlargest(5,"predicted_return_5d").symbol); top_real=set(g.nlargest(5,"actual_return_5d").symbol)
        rows.append({"candidate_month":month,"forecast_origin":origin,"method":method,"eligible_symbols":n,"spearman":_safe_corr(g.predicted_return_5d,g.actual_return_5d,"spearman"),"top5_overlap":len(top_pred&top_real),"predicted_top5_mean_realized_return":g[g.symbol.isin(top_pred)].actual_return_5d.mean(),"ranking_available":True})
    return pd.DataFrame(rows)
def kronos_win_rates(wm):
    kronos=wm[wm.method=="kronos"][["symbol","forecast_origin","final_abs_error"]].rename(columns={"final_abs_error":"kronos_error"}); rows=[]
    for method in sorted(set(wm.method)-{"kronos"}):
        base=wm[wm.method==method][["symbol","forecast_origin","final_abs_error"]].rename(columns={"final_abs_error":"baseline_error"}); merged=kronos.merge(base,on=["symbol","forecast_origin"])
        rows.append({"baseline":method,"defined_windows":len(merged),"kronos_win_rate":float((merged.kronos_error<merged.baseline_error).mean()) if len(merged) else np.nan})
    return rows
def build_overall_summary(window_metrics, skipped_windows, ranking_metrics):
    return {"eligible_windows":int(window_metrics[window_metrics.method=="kronos"].shape[0]),"skipped_windows":int(len(skipped_windows)),"symbols_evaluated":int(window_metrics.symbol.nunique()) if not window_metrics.empty else 0,"ranking_cohorts":int(ranking_metrics.ranking_available.sum()) if not ranking_metrics.empty else 0,"kronos_win_rates":kronos_win_rates(window_metrics)}

from __future__ import annotations
import numpy as np,pandas as pd
def _safe_corr(a,b,method="pearson"):
    f=pd.DataFrame({"a":a,"b":b}).replace([np.inf,-np.inf],np.nan).dropna()
    if len(f)<2 or f.a.nunique()<2 or f.b.nunique()<2:return np.nan
    if method=="spearman":f=f.rank(method="average")
    elif method!="pearson":raise ValueError(f"unsupported correlation method: {method}")
    return float(f.a.corr(f.b,method="pearson"))
def compute_window_metrics(predictions):
    rows=[];keys=["symbol","candidate_month","forecast_origin","method"]
    for key,g in predictions.sort_values("horizon_step").groupby(keys,dropna=False):
        pred=g.predicted_close.to_numpy(float);actual=g.actual_close.to_numpy(float);last=float(g.history_last_close.iloc[0])
        mae=float(np.mean(np.abs(pred-actual)));rmse=float(np.sqrt(np.mean((pred-actual)**2)));fa=float(abs(pred[-1]-actual[-1]));ape=np.nan if actual[-1]==0 else float(fa/abs(actual[-1]));pr=np.nan if last==0 else float(pred[-1]/last-1);ar=np.nan if last==0 else float(actual[-1]/last-1);direction=np.nan if not np.isfinite(pr) or not np.isfinite(ar) else bool(np.sign(pr)==np.sign(ar))
        rows.append(dict(zip(keys,key),mae=mae,rmse=rmse,final_ape=ape,final_abs_error=fa,predicted_return_5d=pr,actual_return_5d=ar,direction_correct=direction))
    return pd.DataFrame(rows)
def aggregate_symbol_metrics(wm):
    rows=[]
    for (s,m),g in wm.groupby(["symbol","method"]):rows.append({"symbol":s,"method":m,"window_count":len(g),"mean_mae":g.mae.mean(),"median_mae":g.mae.median(),"mean_rmse":g.rmse.mean(),"mean_final_ape":g.final_ape.mean(),"direction_accuracy":pd.to_numeric(g.direction_correct,errors="coerce").mean(),"return_pearson":_safe_corr(g.predicted_return_5d,g.actual_return_5d)})
    return pd.DataFrame(rows)
def aggregate_period_metrics(wm):return wm.groupby(["candidate_month","forecast_origin","method"],as_index=False).agg(symbol_count=("symbol","nunique"),mean_mae=("mae","mean"),mean_rmse=("rmse","mean"),direction_accuracy=("direction_correct","mean"))
def compute_ranking_metrics(wm,minimum_cohort):
    rows=[]
    for (month,origin,method),g in wm.groupby(["candidate_month","forecast_origin","method"]):
        g=g.dropna(subset=["predicted_return_5d","actual_return_5d"]);n=g.symbol.nunique()
        if n<minimum_cohort:rows.append({"candidate_month":month,"forecast_origin":origin,"method":method,"eligible_symbols":n,"spearman":np.nan,"top5_overlap":np.nan,"predicted_top5_mean_realized_return":np.nan,"ranking_available":False});continue
        tp=set(g.nlargest(5,"predicted_return_5d").symbol);tr=set(g.nlargest(5,"actual_return_5d").symbol)
        rows.append({"candidate_month":month,"forecast_origin":origin,"method":method,"eligible_symbols":n,"spearman":_safe_corr(g.predicted_return_5d,g.actual_return_5d,"spearman"),"top5_overlap":len(tp&tr),"predicted_top5_mean_realized_return":g[g.symbol.isin(tp)].actual_return_5d.mean(),"ranking_available":True})
    return pd.DataFrame(rows)
def kronos_win_rates(wm):
    k=wm[wm.method=="kronos"][["symbol","forecast_origin","final_abs_error"]].rename(columns={"final_abs_error":"kronos_error"});rows=[]
    for m in sorted(set(wm.method)-{"kronos"}):
        b=wm[wm.method==m][["symbol","forecast_origin","final_abs_error"]].rename(columns={"final_abs_error":"baseline_error"});x=k.merge(b,on=["symbol","forecast_origin"]);rows.append({"baseline":m,"defined_windows":len(x),"kronos_win_rate":float((x.kronos_error<x.baseline_error).mean()) if len(x) else np.nan})
    return rows
def build_overall_summary(wm,skips,ranking):return {"eligible_windows":int((wm.method=="kronos").sum()),"skipped_windows":len(skips),"symbols_evaluated":int(wm.symbol.nunique()) if len(wm) else 0,"ranking_cohorts":int(ranking.ranking_available.sum()) if len(ranking) else 0,"kronos_win_rates":kronos_win_rates(wm)}
BENCHMARK_KEYS=["experiment_arm","symbol","candidate_month","forecast_origin","method"]
def compute_benchmark_window_metrics(predictions):
    rows=[]
    for key,g in predictions.sort_values("horizon_step").groupby(BENCHMARK_KEYS,dropna=False):
        pred=g.predicted_close.to_numpy(float);actual=g.actual_close.to_numpy(float);last=float(g.history_last_close.iloc[0])
        if last<=0 or pred[-1]<=0 or actual[-1]<=0: raise ValueError("log-return prices must be strictly positive")
        pl=float(np.log(pred[-1]/last));al=float(np.log(actual[-1]/last));mae=float(np.mean(np.abs(pred-actual)));rmse=float(np.sqrt(np.mean((pred-actual)**2)));fa=float(abs(pred[-1]-actual[-1]))
        base={"mae":mae,"rmse":rmse,"final_ape":fa/abs(actual[-1]),"final_abs_error":fa,"predicted_return_5d":float(pred[-1]/last-1),"actual_return_5d":float(actual[-1]/last-1),"predicted_log_return_5d":pl,"actual_log_return_5d":al,"log_return_abs_error":abs(pl-al),"direction_correct":bool(np.sign(pl)==np.sign(al))}
        for c in ["final_target_timestamp","history_last_close","actual_final_close","context_view","scoring_target_view","exposure_bucket","context_factor_changed","target_factor_changed","context_max_abs_log_step","target_max_abs_log_from_origin","common_target_fingerprint"]:
            base[c]=g.iloc[-1][c] if c in g.columns else np.nan
        rows.append(dict(zip(BENCHMARK_KEYS,key),**base))
    return pd.DataFrame(rows)
def aggregate_benchmark_symbol_metrics(wm):
    return wm.groupby(["experiment_arm","symbol","method","exposure_bucket"],dropna=False,as_index=False).agg(window_count=("symbol","size"),mean_log_return_abs_error=("log_return_abs_error","mean"),median_log_return_abs_error=("log_return_abs_error","median"),mean_final_ape=("final_ape","mean"),median_final_ape=("final_ape","median"),mean_mae=("mae","mean"),mean_rmse=("rmse","mean"),direction_accuracy=("direction_correct","mean"))
def aggregate_benchmark_period_metrics(wm):
    return wm.groupby(["experiment_arm","candidate_month","forecast_origin","method","exposure_bucket"],dropna=False,as_index=False).agg(symbol_count=("symbol","nunique"),mean_log_return_abs_error=("log_return_abs_error","mean"),direction_accuracy=("direction_correct","mean"))
def compute_benchmark_ranking_metrics(wm,minimum_cohort):
    rows=[]
    for keys,g in wm.groupby(["experiment_arm","candidate_month","forecast_origin","method"],dropna=False):
        n=g.symbol.nunique();arm,month,origin,method=keys
        if n<minimum_cohort:rows.append({"experiment_arm":arm,"candidate_month":month,"forecast_origin":origin,"method":method,"eligible_symbols":n,"spearman":np.nan,"top5_overlap":np.nan,"predicted_top5_mean_realized_return":np.nan,"predicted_top5_mean_realized_log_return":np.nan,"ranking_available":False});continue
        tp=set(g.nlargest(5,"predicted_log_return_5d").symbol);tr=set(g.nlargest(5,"actual_log_return_5d").symbol)
        rows.append({"experiment_arm":arm,"candidate_month":month,"forecast_origin":origin,"method":method,"eligible_symbols":n,"spearman":_safe_corr(g.predicted_log_return_5d,g.actual_log_return_5d,"spearman"),"top5_overlap":len(tp&tr),"predicted_top5_mean_realized_return":g[g.symbol.isin(tp)].actual_return_5d.mean(),"predicted_top5_mean_realized_log_return":g[g.symbol.isin(tp)].actual_log_return_5d.mean(),"ranking_available":True})
    return pd.DataFrame(rows)

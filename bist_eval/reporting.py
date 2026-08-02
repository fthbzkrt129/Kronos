from __future__ import annotations
from datetime import date,datetime
from pathlib import Path
import json,os
import numpy as np,pandas as pd
PREDICTION_COLUMNS=["symbol","candidate_month","forecast_origin","target_timestamp","horizon_step","method","predicted_close","actual_close","history_last_close"]
WINDOW_METRIC_COLUMNS=["symbol","candidate_month","forecast_origin","method","mae","rmse","final_ape","final_abs_error","predicted_return_5d","actual_return_5d","direction_correct"]
SKIP_COLUMNS=["symbol","candidate_month","reason_code","reason_detail","available_history_rows","available_target_rows"]
BENCHMARK_PREDICTION_COLUMNS=["experiment_arm","symbol","candidate_month","forecast_origin","target_timestamp","horizon_step","method","predicted_close","actual_close","history_last_close","context_view","scoring_target_view","exposure_bucket","context_factor_changed","target_factor_changed","context_max_abs_log_step","target_max_abs_log_from_origin","common_target_fingerprint"]
BENCHMARK_WINDOW_METRIC_COLUMNS=["experiment_arm","symbol","candidate_month","forecast_origin","method","mae","rmse","final_ape","final_abs_error","predicted_return_5d","actual_return_5d","predicted_log_return_5d","actual_log_return_5d","log_return_abs_error","direction_correct","final_target_timestamp","history_last_close","actual_final_close","context_view","scoring_target_view","exposure_bucket","context_factor_changed","target_factor_changed","context_max_abs_log_step","target_max_abs_log_from_origin","common_target_fingerprint"]
def _json_default(v):
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.floating):return None if np.isnan(v) else float(v)
    if isinstance(v,(pd.Timestamp,datetime,date)):return v.isoformat()
    if isinstance(v,np.bool_):return bool(v)
    if pd.isna(v):return None
    raise TypeError(type(v).__name__)
def _atomic_text(text,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+".tmp");tmp.write_text(text,encoding="utf-8");os.replace(tmp,p)
def _atomic_csv(df,path,columns=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);out=df.copy()
    if columns is not None:
        missing=set(columns)-set(out.columns)
        if missing:raise ValueError("missing output columns: "+", ".join(sorted(missing)))
        out=out.loc[:,columns]
    for c in out.columns:
        if "timestamp" in c or c=="forecast_origin":out[c]=out[c].map(lambda x:pd.Timestamp(x).isoformat() if not pd.isna(x) else x)
    tmp=p.with_suffix(p.suffix+".tmp");out.to_csv(tmp,index=False);os.replace(tmp,p)
def _atomic_json(payload,path):_atomic_text(json.dumps(payload,ensure_ascii=False,indent=2,default=_json_default)+"\n",path)
def validate_prediction_keys(pred):
    if pred.duplicated(["symbol","forecast_origin","target_timestamp","method"]).any():raise ValueError("duplicate prediction key")
def validate_benchmark_prediction_keys(pred):
    if pred.duplicated(["experiment_arm","symbol","forecast_origin","target_timestamp","method"]).any():raise ValueError("duplicate benchmark prediction key")
def write_shard_output(output_dir,predictions,window_metrics,skips,manifest):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);(out/"COMPLETED").unlink(missing_ok=True);validate_prediction_keys(predictions)
    _atomic_csv(predictions,out/"predictions.csv",PREDICTION_COLUMNS);_atomic_csv(window_metrics,out/"window_metrics.csv",WINDOW_METRIC_COLUMNS);_atomic_csv(skips,out/"skipped_windows.csv",SKIP_COLUMNS);_atomic_json(manifest,out/"shard_manifest.json");_atomic_text("ok\n",out/"COMPLETED")
def write_benchmark_shard_output(output_dir,predictions,window_metrics,skips,manifest):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);(out/"COMPLETED").unlink(missing_ok=True);validate_benchmark_prediction_keys(predictions)
    _atomic_csv(predictions,out/"predictions.csv",BENCHMARK_PREDICTION_COLUMNS);_atomic_csv(window_metrics,out/"window_metrics.csv",BENCHMARK_WINDOW_METRIC_COLUMNS);_atomic_csv(skips,out/"skipped_windows.csv",SKIP_COLUMNS);_atomic_json(manifest,out/"shard_manifest.json");_atomic_text("ok\n",out/"COMPLETED")
def _report(summary):return f"# Zero-shot historical evaluation of the 2026 Q3 BIST 100 constituent snapshot over 2023-2026\n\n- Eligible windows: {summary.get('eligible_windows',0)}\n\nThis is research-only output and not investment advice. The current constituent snapshot creates survivorship bias. Yahoo is not an official licensed Borsa Istanbul feed.\n"
def write_reduced_output(output_dir,*,predictions,window_metrics,skips,symbol_metrics,period_metrics,ranking_metrics,summary,manifest):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);(out/"COMPLETED").unlink(missing_ok=True);validate_prediction_keys(predictions)
    _atomic_csv(predictions,out/"predictions.csv",PREDICTION_COLUMNS);_atomic_csv(window_metrics,out/"window_metrics.csv",WINDOW_METRIC_COLUMNS);_atomic_csv(skips,out/"skipped_windows.csv",SKIP_COLUMNS);_atomic_csv(symbol_metrics,out/"symbol_metrics.csv");_atomic_csv(period_metrics,out/"period_metrics.csv");_atomic_csv(ranking_metrics,out/"ranking_metrics.csv");_atomic_json(summary,out/"summary.json");_atomic_json(manifest,out/"run_manifest.json");_atomic_text(_report(summary),out/"report.md");_atomic_text("ok\n",out/"COMPLETED")
def _benchmark_report(summary):
    q=summary.get("questions",{})
    lines=["# Paired adjusted-price zero-shot benchmark of Kronos-mini and Kronos-small on the 2026 Q3 BIST 100 constituent snapshot over 2023-2026","","## Technical completion",f"- Prediction rows: {summary.get('prediction_rows',0)}",f"- Eligible model windows: {summary.get('eligible_model_windows',0)}","","## Benchmark questions"]
    for k,v in q.items():lines.append(f"- {k}: {v}")
    lines.extend(["","## Limitations","This is research-only output, not investment advice or a trading strategy. The 2026 Q3 constituent snapshot is projected backward and introduces survivorship and selection bias. Yahoo is not an official licensed Borsa Istanbul source. The provider factor-ratio method is a research assumption. Volume remains raw, amount is estimated, and no transaction costs, liquidity constraints, portfolio construction, broker connection, or order placement are included."])
    return "\n".join(lines)+"\n"
def write_benchmark_reduced_output(output_dir,*,factor_manifest,predictions,window_metrics,skips,symbol_metrics,period_metrics,ranking_metrics,paired_comparisons,bootstrap_intervals,summary,manifest):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);(out/"COMPLETED").unlink(missing_ok=True);validate_benchmark_prediction_keys(predictions)
    _atomic_json(factor_manifest,out/"adjusted_data_manifest.json");_atomic_csv(predictions,out/"predictions.csv",BENCHMARK_PREDICTION_COLUMNS);_atomic_csv(window_metrics,out/"window_metrics.csv",BENCHMARK_WINDOW_METRIC_COLUMNS);_atomic_csv(skips,out/"skipped_windows.csv",SKIP_COLUMNS);_atomic_csv(symbol_metrics,out/"symbol_metrics.csv");_atomic_csv(period_metrics,out/"period_metrics.csv");_atomic_csv(ranking_metrics,out/"ranking_metrics.csv");_atomic_csv(paired_comparisons,out/"paired_comparisons.csv");_atomic_csv(bootstrap_intervals,out/"bootstrap_intervals.csv");_atomic_json(summary,out/"summary.json");_atomic_json(manifest,out/"run_manifest.json");_atomic_text(_benchmark_report(summary),out/"report.md");_atomic_text("ok\n",out/"COMPLETED")

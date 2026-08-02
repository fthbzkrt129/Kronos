"""Atomic, schema-checked evaluation artifacts."""
from __future__ import annotations
from datetime import date, datetime
from pathlib import Path
import json, os
import numpy as np, pandas as pd
PREDICTION_COLUMNS=["symbol","candidate_month","forecast_origin","target_timestamp","horizon_step","method","predicted_close","actual_close","history_last_close"]
WINDOW_METRIC_COLUMNS=["symbol","candidate_month","forecast_origin","method","mae","rmse","final_ape","final_abs_error","predicted_return_5d","actual_return_5d","direction_correct"]
SKIP_COLUMNS=["symbol","candidate_month","reason_code","reason_detail","available_history_rows","available_target_rows"]
def _json_default(v):
    if isinstance(v,np.integer): return int(v)
    if isinstance(v,np.floating): return None if np.isnan(v) else float(v)
    if isinstance(v,(pd.Timestamp,datetime,date)): return v.isoformat()
    if pd.isna(v): return None
    raise TypeError(type(v).__name__)
def _atomic_text(text,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(text,encoding="utf-8"); os.replace(tmp,path)
def _atomic_csv(df,path,columns=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); out=df.copy()
    if columns is not None:
        missing=set(columns)-set(out.columns)
        if missing: raise ValueError("missing output columns: "+", ".join(sorted(missing)))
        out=out.loc[:,columns]
    for c in out.columns:
        if "timestamp" in c or c=="forecast_origin": out[c]=out[c].map(lambda x: pd.Timestamp(x).isoformat() if not pd.isna(x) else x)
    tmp=path.with_suffix(path.suffix+".tmp"); out.to_csv(tmp,index=False); os.replace(tmp,path)
def _atomic_json(payload,path): _atomic_text(json.dumps(payload,ensure_ascii=False,indent=2,default=_json_default)+"\n",path)
def validate_prediction_keys(pred):
    if pred.duplicated(["symbol","forecast_origin","target_timestamp","method"]).any(): raise ValueError("duplicate prediction key")
def write_shard_output(output_dir,predictions,window_metrics,skips,manifest):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); (out/"COMPLETED").unlink(missing_ok=True); validate_prediction_keys(predictions)
    _atomic_csv(predictions,out/"predictions.csv",PREDICTION_COLUMNS); _atomic_csv(window_metrics,out/"window_metrics.csv",WINDOW_METRIC_COLUMNS); _atomic_csv(skips,out/"skipped_windows.csv",SKIP_COLUMNS); _atomic_json(manifest,out/"shard_manifest.json"); _atomic_text("ok\n",out/"COMPLETED")
def _report(summary):
    return f"""# Zero-shot historical evaluation of the 2026 Q3 BIST 100 constituent snapshot over 2023-2026

## Summary

- Eligible windows: {summary.get('eligible_windows',0)}
- Skipped windows: {summary.get('skipped_windows',0)}
- Symbols evaluated: {summary.get('symbols_evaluated',0)}

## Limitations

This is research-only output and not investment advice. The 2026 Q3 constituent snapshot is projected backward, creating survivorship and selection bias. Yahoo is not an official licensed Borsa Istanbul feed. Predicted-top-five realized return is an uncosted diagnostic, not a tradable strategy return.
"""
def write_reduced_output(output_dir,*,predictions,window_metrics,skips,symbol_metrics,period_metrics,ranking_metrics,summary,manifest):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); (out/"COMPLETED").unlink(missing_ok=True); validate_prediction_keys(predictions)
    _atomic_csv(predictions,out/"predictions.csv",PREDICTION_COLUMNS); _atomic_csv(window_metrics,out/"window_metrics.csv",WINDOW_METRIC_COLUMNS); _atomic_csv(skips,out/"skipped_windows.csv",SKIP_COLUMNS); _atomic_csv(symbol_metrics,out/"symbol_metrics.csv"); _atomic_csv(period_metrics,out/"period_metrics.csv"); _atomic_csv(ranking_metrics,out/"ranking_metrics.csv"); _atomic_json(summary,out/"summary.json"); _atomic_json(manifest,out/"run_manifest.json"); _atomic_text(_report(summary),out/"report.md"); _atomic_text("ok\n",out/"COMPLETED")

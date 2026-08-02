#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
if __package__ in {None,""}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bist_eval.comparison import DEFAULT_COMPARISONS,build_paired_comparison,summarize_comparison_intervals
from bist_eval.metrics import aggregate_benchmark_period_metrics,aggregate_benchmark_symbol_metrics,compute_benchmark_ranking_metrics
from bist_eval.reporting import validate_benchmark_prediction_keys,write_benchmark_reduced_output
from bist_eval.sharding import validate_benchmark_shard_manifests
def _read_shards(root):
    dirs=sorted([p for p in Path(root).iterdir() if p.is_dir()]);records=[]
    for d in dirs:
        if not (d/"COMPLETED").is_file():raise ValueError(f"shard missing COMPLETED: {d}")
        if not (d/"shard_manifest.json").is_file():continue
        m=json.loads((d/"shard_manifest.json").read_text());records.append((m,d))
    records.sort(key=lambda x:x[0]["shard_index"]);return records
def run_adjusted_reducer(*,mini_shards_dir,small_shards_dir,expected_shards,factor_manifest,output_dir,minimum_ranking_cohort=20,bootstrap_draws=10000,bootstrap_confidence=.95,bootstrap_seed=20260802):
    mini=_read_shards(mini_shards_dir);small=_read_shards(small_shards_dir);validate_benchmark_shard_manifests([m for m,d in mini],[m for m,d in small],expected_shards)
    predictions=[];window_metrics=[];skips=[]
    for m,d in [*mini,*small]:
        predictions.append(pd.read_csv(d/"predictions.csv"));window_metrics.append(pd.read_csv(d/"window_metrics.csv"));skips.append(pd.read_csv(d/"skipped_windows.csv"))
    pred=pd.concat(predictions,ignore_index=True);wm=pd.concat(window_metrics,ignore_index=True);sk=pd.concat(skips,ignore_index=True);validate_benchmark_prediction_keys(pred)
    for c in ("forecast_origin","target_timestamp"):pred[c]=pd.to_datetime(pred[c])
    wm.forecast_origin=pd.to_datetime(wm.forecast_origin)
    symbol=aggregate_benchmark_symbol_metrics(wm);period=aggregate_benchmark_period_metrics(wm);ranking=compute_benchmark_ranking_metrics(wm,minimum_ranking_cohort)
    paired_parts=[];interval_parts=[];decisions={}
    for spec in DEFAULT_COMPARISONS:
        paired=build_paired_comparison(wm,spec)
        if len(paired)==0:decisions[spec.comparison_id]="unavailable";continue
        paired_parts.append(paired);intervals=summarize_comparison_intervals(paired,draws=bootstrap_draws,confidence=bootstrap_confidence,seed=bootstrap_seed);interval_parts.append(intervals);decisions[spec.comparison_id]=intervals.decision.iloc[0]
    paired_all=pd.concat(paired_parts,ignore_index=True) if paired_parts else pd.DataFrame();interval_all=pd.concat(interval_parts,ignore_index=True) if interval_parts else pd.DataFrame()
    arm_summary=wm.groupby(["experiment_arm","method"],as_index=False).agg(windows=("symbol","size"),mean_log_return_abs_error=("log_return_abs_error","mean"),median_log_return_abs_error=("log_return_abs_error","median"),direction_accuracy=("direction_correct","mean"))
    questions={"Does adjusted data improve Kronos-mini?":decisions.get("adjusted-mini_vs_raw-mini","unavailable"),"Does Kronos-small outperform adjusted Kronos-mini?":decisions.get("adjusted-small_vs_adjusted-mini","unavailable"),"Does adjusted Kronos-mini beat last-close?":decisions.get("adjusted-mini_vs_last-close","unavailable"),"Does adjusted Kronos-small beat last-close?":decisions.get("adjusted-small_vs_last-close","unavailable"),"Is cross-sectional ranking reliable?":"review ranking_metrics.csv"}
    summary={"prediction_rows":len(pred),"eligible_model_windows":int(((wm.method=="kronos")&wm.experiment_arm.isin(["raw-mini","adjusted-mini","adjusted-small"])).sum()),"symbols_evaluated":int(wm.symbol.nunique()),"questions":questions,"arm_metrics":arm_summary.to_dict("records"),"comparison_decisions":decisions,"exposure_counts":wm.groupby("exposure_bucket").size().to_dict()}
    first=mini[0][0];manifest={"schema_version":1,"expected_shards":expected_shards,"source_data_fingerprint":first["source_data_fingerprint"],"factor_fingerprint":first["factor_fingerprint"],"universe_fingerprint":first["universe_fingerprint"],"cohort_fingerprint":first["cohort_fingerprint"],"common_target_fingerprint":first["common_target_fingerprint"],"common_protocol_fingerprint":first["common_protocol_fingerprint"],"mini_model_revision":mini[0][0].get("model_revision"),"mini_tokenizer_revision":mini[0][0].get("tokenizer_revision"),"small_model_revision":small[0][0].get("model_revision"),"small_tokenizer_revision":small[0][0].get("tokenizer_revision"),"symbols":sorted(set(wm.symbol)),"bootstrap":{"draws":bootstrap_draws,"confidence":bootstrap_confidence,"seed":bootstrap_seed}}
    factor=json.loads(Path(factor_manifest).read_text());write_benchmark_reduced_output(output_dir,factor_manifest=factor,predictions=pred,window_metrics=wm,skips=sk,symbol_metrics=symbol,period_metrics=period,ranking_metrics=ranking,paired_comparisons=paired_all,bootstrap_intervals=interval_all,summary=summary,manifest=manifest);return summary
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--mini-shards-dir",type=Path,required=True);p.add_argument("--small-shards-dir",type=Path,required=True);p.add_argument("--expected-shards",type=int,required=True);p.add_argument("--factor-manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--minimum-ranking-cohort",type=int,default=20);p.add_argument("--bootstrap-draws",type=int,default=10000);p.add_argument("--bootstrap-confidence",type=float,default=.95);p.add_argument("--bootstrap-seed",type=int,default=20260802);a=p.parse_args(argv)
    print(json.dumps(run_adjusted_reducer(mini_shards_dir=a.mini_shards_dir,small_shards_dir=a.small_shards_dir,expected_shards=a.expected_shards,factor_manifest=a.factor_manifest,output_dir=a.output,minimum_ranking_cohort=a.minimum_ranking_cohort,bootstrap_draws=a.bootstrap_draws,bootstrap_confidence=a.bootstrap_confidence,bootstrap_seed=a.bootstrap_seed),indent=2));return 0
if __name__=="__main__":raise SystemExit(main())

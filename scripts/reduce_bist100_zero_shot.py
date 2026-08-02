#!/usr/bin/env python3
"""Reduce compatible BIST zero-shot shard artifacts into final reports."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
if __package__ in {None,""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bist_eval.metrics import aggregate_period_metrics,aggregate_symbol_metrics,build_overall_summary,compute_ranking_metrics
from bist_eval.reporting import write_reduced_output
from bist_eval.sharding import validate_shard_manifests
def run_reducer(*,shards_dir,expected_shards,output_dir,minimum_ranking_cohort=20):
    root=Path(shards_dir); dirs=sorted([p for p in root.iterdir() if p.is_dir()]); manifests=[]; pred=[]; wm=[]; skips=[]
    for d in dirs:
        if not (d/"COMPLETED").is_file(): raise ValueError(f"shard missing COMPLETED: {d}")
        if (d/"shard_manifest.json").is_file(): manifests.append(json.loads((d/"shard_manifest.json").read_text()))
    validate_shard_manifests(manifests,expected_shards)
    for idx in range(expected_shards):
        candidates=[d for d in dirs if (d/"shard_manifest.json").is_file() and json.loads((d/"shard_manifest.json").read_text()).get("shard_index")==idx]
        if len(candidates)!=1: raise ValueError("unable to map shard directory")
        d=candidates[0]; pred.append(pd.read_csv(d/"predictions.csv")); wm.append(pd.read_csv(d/"window_metrics.csv")); skips.append(pd.read_csv(d/"skipped_windows.csv"))
    predictions=pd.concat(pred,ignore_index=True); window_metrics=pd.concat(wm,ignore_index=True); skipped=pd.concat(skips,ignore_index=True)
    predictions["forecast_origin"]=pd.to_datetime(predictions.forecast_origin); predictions["target_timestamp"]=pd.to_datetime(predictions.target_timestamp); window_metrics["forecast_origin"]=pd.to_datetime(window_metrics.forecast_origin)
    symbol_metrics=aggregate_symbol_metrics(window_metrics); period_metrics=aggregate_period_metrics(window_metrics); ranking=compute_ranking_metrics(window_metrics,minimum_ranking_cohort); summary=build_overall_summary(window_metrics,skipped,ranking)
    run_manifest={"schema_version":1,"expected_shards":expected_shards,"config_fingerprint":manifests[0]["config_fingerprint"],"source_data_fingerprint":manifests[0]["source_data_fingerprint"],"model_revision":manifests[0].get("model_revision"),"tokenizer_revision":manifests[0].get("tokenizer_revision"),"symbols":sorted(s for m in manifests for s in m.get("symbols",[]))}
    write_reduced_output(output_dir,predictions=predictions,window_metrics=window_metrics,skips=skipped,symbol_metrics=symbol_metrics,period_metrics=period_metrics,ranking_metrics=ranking,summary=summary,manifest=run_manifest); return summary
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--shards-dir",type=Path,required=True); p.add_argument("--expected-shards",type=int,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--minimum-ranking-cohort",type=int,default=20); a=p.parse_args(argv)
    print(json.dumps(run_reducer(shards_dir=a.shards_dir,expected_shards=a.expected_shards,output_dir=a.output,minimum_ranking_cohort=a.minimum_ranking_cohort),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate one BIST zero-shot symbol subset or deterministic shard."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict
from pathlib import Path
from collections import defaultdict
import pandas as pd
if __package__ in {None,""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bist_data.universe import load_universe
from bist_eval.baselines import forecast_baselines
from bist_eval.calendar import build_canonical_calendar, build_monthly_cohorts
from bist_eval.config import EvaluationConfig
from bist_eval.data import discover_symbol_files, load_symbol_frame, load_timestamp_coverage, source_data_fingerprint
from bist_eval.metrics import compute_window_metrics
from bist_eval.model_adapter import KronosModelAdapter
from bist_eval.reporting import PREDICTION_COLUMNS, SKIP_COLUMNS, write_shard_output
from bist_eval.sharding import select_shard
from bist_eval.windows import build_symbol_windows

def _parse_symbols(values):
    if not values:return None
    out=[]
    for value in values: out.extend(x.strip().upper() for x in value.split(",") if x.strip())
    return list(dict.fromkeys(out))
def run_evaluation(*,data_dir,universe_path,output_dir,config:EvaluationConfig,symbols=None,shard_index=None,strict=False,model_adapter=None,source_manifest=None):
    entries=load_universe(universe_path); ordered=[e.symbol for e in entries]; requested=_parse_symbols(symbols)
    if requested is not None and shard_index is not None: raise ValueError("--symbols and --shard-index are mutually exclusive")
    all_files=discover_symbol_files(Path(data_dir),ordered); missing=sorted(set(ordered)-set(all_files))
    if strict and missing: raise ValueError("missing symbol files: "+", ".join(missing))
    coverage=load_timestamp_coverage(all_files)
    calendar=build_canonical_calendar(coverage,coverage_threshold=config.calendar_coverage,start_date=config.start_date,end_date=config.end_date)
    cohorts,calendar_skips=build_monthly_cohorts(calendar,horizon=config.horizon)
    if requested is not None:
        unknown=sorted(set(requested)-set(ordered))
        if unknown: raise ValueError("unknown symbols: "+", ".join(unknown))
        selected=tuple(requested); effective_shard_index=0; effective_shard_count=1
    elif shard_index is not None:
        selected=select_shard(ordered,config.shard_count,shard_index); effective_shard_index=shard_index; effective_shard_count=config.shard_count
    else:
        selected=tuple(ordered); effective_shard_index=0; effective_shard_count=1
    windows=[]; skip_rows=[]
    for symbol in selected:
        path=all_files.get(symbol)
        if path is None:
            skip_rows.append({"symbol":symbol,"candidate_month":"*","reason_code":"missing_symbol_file","reason_detail":"source CSV missing","available_history_rows":0,"available_target_rows":0}); continue
        try: frame=load_symbol_frame(path)
        except Exception as exc:
            if strict: raise
            skip_rows.append({"symbol":symbol,"candidate_month":"*","reason_code":"invalid_symbol_file","reason_detail":str(exc),"available_history_rows":0,"available_target_rows":0}); continue
        w,s=build_symbol_windows(symbol,frame,cohorts,lookback=config.lookback,horizon=config.horizon); windows.extend(w); skip_rows.extend(asdict(x) for x in s)
    for cs in calendar_skips: skip_rows.append({"symbol":"*","candidate_month":cs.candidate_month,"reason_code":cs.reason_code,"reason_detail":cs.reason_detail,"available_history_rows":0,"available_target_rows":0})
    adapter=model_adapter or KronosModelAdapter(model_id=config.model_id,tokenizer_id=config.tokenizer_id,model_revision=config.model_revision,tokenizer_revision=config.tokenizer_revision,temperature=config.temperature,top_p=config.top_p,sample_count=config.sample_count,seed=config.seed)
    prediction_rows=[]; by_cohort=defaultdict(list)
    for w in windows: by_cohort[(w.forecast_origin,w.target_timestamps)].append(w)
    for _,group in sorted(by_cohort.items(), key=lambda kv:kv[0][0]):
        model_outputs=adapter.predict_cohort(group)
        for w in group:
            methods=forecast_baselines(w.context,config.horizon); methods["kronos"]=model_outputs[w.symbol]
            actual=w.target.close.to_numpy(float); last=float(w.context.close.iloc[-1])
            for method,pred in methods.items():
                for step,(ts,p,a) in enumerate(zip(w.target_timestamps,pred,actual),1): prediction_rows.append({"symbol":w.symbol,"candidate_month":w.candidate_month,"forecast_origin":w.forecast_origin,"target_timestamp":ts,"horizon_step":step,"method":method,"predicted_close":float(p),"actual_close":float(a),"history_last_close":last})
    predictions=pd.DataFrame(prediction_rows,columns=PREDICTION_COLUMNS)
    window_metrics=compute_window_metrics(predictions) if len(predictions) else pd.DataFrame(columns=["symbol","candidate_month","forecast_origin","method","mae","rmse","final_ape","final_abs_error","predicted_return_5d","actual_return_5d","direction_correct"])
    skips=pd.DataFrame(skip_rows,columns=SKIP_COLUMNS); fingerprint=source_data_fingerprint(all_files,Path(source_manifest) if source_manifest else None)
    manifest={"schema_version":1,"shard_index":effective_shard_index,"shard_count":effective_shard_count,"symbols":list(selected),"config":config.to_canonical_dict(),"config_fingerprint":config.fingerprint,"source_data_fingerprint":fingerprint,"model_revision":config.model_revision,"tokenizer_revision":config.tokenizer_revision,"prediction_rows":len(predictions),"eligible_windows":int((window_metrics.method=="kronos").sum()) if len(window_metrics) else 0,"skipped_windows":len(skips)}
    write_shard_output(output_dir,predictions,window_metrics,skips,manifest); return manifest
def build_parser():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,required=True); p.add_argument("--universe",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--start",default="2023-01-01"); p.add_argument("--end",default="2026-08-02"); p.add_argument("--lookback",type=int,default=400); p.add_argument("--horizon",type=int,default=5); p.add_argument("--calendar-coverage",type=float,default=.8); p.add_argument("--minimum-ranking-cohort",type=int,default=20)
    p.add_argument("--model-id",default="NeoQuasar/Kronos-mini"); p.add_argument("--tokenizer-id",default="NeoQuasar/Kronos-Tokenizer-2k"); p.add_argument("--model-revision"); p.add_argument("--tokenizer-revision"); p.add_argument("--temperature",type=float,default=1.0); p.add_argument("--top-p",type=float,default=.9); p.add_argument("--sample-count",type=int,default=1); p.add_argument("--seed",type=int,default=20260802); p.add_argument("--device"); p.add_argument("--symbols",nargs="+"); p.add_argument("--shard-index",type=int); p.add_argument("--shard-count",type=int,default=10); p.add_argument("--strict",action="store_true"); p.add_argument("--source-manifest",type=Path); p.add_argument("--model-path",type=Path); p.add_argument("--tokenizer-path",type=Path); return p
def main(argv=None):
    args=build_parser().parse_args(argv)
    cfg=EvaluationConfig(start_date=args.start,end_date=args.end,lookback=args.lookback,horizon=args.horizon,calendar_coverage=args.calendar_coverage,minimum_ranking_cohort=args.minimum_ranking_cohort,model_id=args.model_id,tokenizer_id=args.tokenizer_id,model_revision=args.model_revision,tokenizer_revision=args.tokenizer_revision,temperature=args.temperature,top_p=args.top_p,sample_count=args.sample_count,seed=args.seed,shard_count=args.shard_count)
    adapter=KronosModelAdapter(model_id=args.model_id,tokenizer_id=args.tokenizer_id,model_revision=args.model_revision,tokenizer_revision=args.tokenizer_revision,model_path=args.model_path,tokenizer_path=args.tokenizer_path,device=args.device,temperature=args.temperature,top_p=args.top_p,sample_count=args.sample_count,seed=args.seed)
    try: manifest=run_evaluation(data_dir=args.data_dir,universe_path=args.universe,output_dir=args.output,config=cfg,symbols=args.symbols,shard_index=args.shard_index,strict=args.strict,model_adapter=adapter,source_manifest=args.source_manifest)
    except ValueError as exc: raise SystemExit(str(exc))
    print(json.dumps(manifest,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())

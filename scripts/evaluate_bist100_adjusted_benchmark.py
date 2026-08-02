#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
if __package__ in {None,""}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bist_data.universe import load_universe
from bist_eval.benchmark import run_mini_pair_shard,run_small_shard
from bist_eval.calendar import build_canonical_calendar,build_monthly_cohorts
from bist_eval.config import AdjustedBenchmarkConfig
from bist_eval.data import discover_raw_symbol_files,load_raw_symbol_frame,load_timestamp_coverage,source_data_fingerprint
from bist_eval.model_adapter import KronosModelAdapter
from bist_eval.reporting import write_benchmark_shard_output
from bist_eval.sharding import select_shard
def _parse(values):
    if not values:return None
    out=[]
    for v in values:out.extend(x.strip().upper() for x in v.split(",") if x.strip())
    return list(dict.fromkeys(out))
def _hash(payload):return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def run_adjusted_evaluation(*,mode,raw_dir,source_manifest,factor_manifest,universe_path,output_dir,config,symbols=None,shard_index=None,strict=False,adapter=None):
    entries=load_universe(universe_path);ordered=[e.symbol for e in entries];requested=_parse(symbols)
    if requested is not None and shard_index is not None:raise ValueError("--symbols and --shard-index are mutually exclusive")
    all_files=discover_raw_symbol_files(raw_dir,ordered);missing=sorted(set(ordered)-set(all_files))
    if strict and missing:raise ValueError("missing raw symbol files: "+", ".join(missing))
    factor=json.loads(Path(factor_manifest).read_text())
    if not factor.get("static_adjusted_model_inputs_written") is False:raise ValueError("invalid factor manifest")
    if factor.get("formula_version")!=config.adjustment_formula_version:raise ValueError("factor formula mismatch")
    if abs(float(factor.get("material_factor_tolerance"))-config.material_factor_tolerance)>0:raise ValueError("factor tolerance mismatch")
    current_source_digest=hashlib.sha256(Path(source_manifest).read_bytes()).hexdigest()
    if factor.get("source_manifest_digest")!=current_source_digest:raise ValueError("factor/source manifest digest mismatch")
    current_universe_digest=hashlib.sha256(Path(universe_path).read_bytes()).hexdigest()
    if factor.get("universe_digest")!=current_universe_digest:raise ValueError("factor/universe digest mismatch")
    coverage=load_timestamp_coverage(all_files);calendar=build_canonical_calendar(coverage,coverage_threshold=config.calendar_coverage,start_date=config.start_date,end_date=config.end_date);cohorts,calendar_skips=build_monthly_cohorts(calendar,horizon=config.horizon)
    if requested is not None:
        unknown=sorted(set(requested)-set(ordered))
        if unknown:raise ValueError("unknown symbols: "+", ".join(unknown))
        selected=tuple(requested);effective_index=0;effective_count=1
    elif shard_index is not None:selected=select_shard(ordered,config.shard_count,shard_index);effective_index=shard_index;effective_count=config.shard_count
    else:selected=tuple(ordered);effective_index=0;effective_count=1
    raw_frames={s:load_raw_symbol_frame(all_files[s]) for s in selected if s in all_files}
    cohort_payload=[{"month":c.candidate_month,"origin":c.forecast_origin.isoformat(),"targets":[x.isoformat() for x in c.target_timestamps]} for c in cohorts];cohort_fp=_hash(cohort_payload);target_fp=_hash({"formula":config.adjustment_formula_version,"target":config.scoring_target_view,"cohorts":cohort_fp})
    source_fp=source_data_fingerprint(all_files,Path(source_manifest) if source_manifest else None);universe_fp=hashlib.sha256(Path(universe_path).read_bytes()).hexdigest()
    manifest_base={"schema_version":1,"shard_index":effective_index,"shard_count":effective_count,"source_data_fingerprint":source_fp,"factor_fingerprint":factor["aggregate_factor_fingerprint"],"universe_fingerprint":universe_fp,"cohort_fingerprint":cohort_fp,"common_target_fingerprint":target_fp,"common_protocol_fingerprint":config.common_protocol_fingerprint,"config_fingerprint":config.fingerprint,"model_id":config.model_id,"tokenizer_id":config.tokenizer_id,"model_revision":config.model_revision,"tokenizer_revision":config.tokenizer_revision,"calendar_skip_count":len(calendar_skips)}
    adapter=adapter or KronosModelAdapter(model_id=config.model_id,tokenizer_id=config.tokenizer_id,model_revision=config.model_revision,tokenizer_revision=config.tokenizer_revision,temperature=config.temperature,top_p=config.top_p,sample_count=config.sample_count,seed=config.seed)
    factor_symbols={item["symbol"] for item in factor.get("symbols",[])}
    missing_factor_symbols=sorted(set(selected)-factor_symbols)
    if missing_factor_symbols:raise ValueError("factor manifest missing symbols: "+", ".join(missing_factor_symbols))
    if mode=="mini-pair":result=run_mini_pair_shard(raw_frames=raw_frames,cohorts=cohorts,symbols=selected,config=config,adapter=adapter,manifest_base=manifest_base)
    elif mode=="small":result=run_small_shard(raw_frames=raw_frames,cohorts=cohorts,symbols=selected,config=config,adapter=adapter,manifest_base=manifest_base)
    else:raise ValueError("unsupported mode")
    if calendar_skips:
        import pandas as pd
        calendar_rows=pd.DataFrame([{"symbol":"*","candidate_month":x.candidate_month,"reason_code":x.reason_code,"reason_detail":x.reason_detail,"available_history_rows":0,"available_target_rows":0} for x in calendar_skips])
        result_skips=pd.concat([result.skips,calendar_rows],ignore_index=True)
    else:result_skips=result.skips
    write_benchmark_shard_output(output_dir,result.predictions,result.window_metrics,result_skips,result.manifest);return result.manifest
def build_parser():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["mini-pair","small"],required=True);p.add_argument("--raw-dir",type=Path,required=True);p.add_argument("--source-manifest",type=Path,required=True);p.add_argument("--factor-manifest",type=Path,required=True);p.add_argument("--universe",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--start",default="2023-01-01");p.add_argument("--end",default="2026-08-02");p.add_argument("--lookback",type=int,default=400);p.add_argument("--horizon",type=int,default=5);p.add_argument("--calendar-coverage",type=float,default=.8);p.add_argument("--minimum-ranking-cohort",type=int,default=20);p.add_argument("--material-factor-tolerance",type=float,default=1e-8)
    p.add_argument("--model-id");p.add_argument("--tokenizer-id");p.add_argument("--model-path",type=Path);p.add_argument("--tokenizer-path",type=Path);p.add_argument("--model-revision");p.add_argument("--tokenizer-revision");p.add_argument("--temperature",type=float,default=1.);p.add_argument("--top-p",type=float,default=.9);p.add_argument("--sample-count",type=int,default=1);p.add_argument("--seed",type=int,default=20260802);p.add_argument("--device");p.add_argument("--symbols",nargs="+");p.add_argument("--shard-index",type=int);p.add_argument("--shard-count",type=int,default=10);p.add_argument("--strict",action="store_true");return p
def main(argv=None):
    a=build_parser().parse_args(argv);arm="adjusted-mini" if a.mode=="mini-pair" else "adjusted-small";model=a.model_id or ("NeoQuasar/Kronos-mini" if a.mode=="mini-pair" else "NeoQuasar/Kronos-small");tokenizer=a.tokenizer_id or ("NeoQuasar/Kronos-Tokenizer-2k" if a.mode=="mini-pair" else "NeoQuasar/Kronos-Tokenizer-base")
    cfg=AdjustedBenchmarkConfig(experiment_arm=arm,context_view="origin_rebased",model_id=model,tokenizer_id=tokenizer,model_revision=a.model_revision,tokenizer_revision=a.tokenizer_revision,start_date=a.start,end_date=a.end,lookback=a.lookback,horizon=a.horizon,calendar_coverage=a.calendar_coverage,minimum_ranking_cohort=a.minimum_ranking_cohort,material_factor_tolerance=a.material_factor_tolerance,temperature=a.temperature,top_p=a.top_p,sample_count=a.sample_count,seed=a.seed,shard_count=a.shard_count)
    adapter=KronosModelAdapter(model_id=model,tokenizer_id=tokenizer,model_revision=a.model_revision,tokenizer_revision=a.tokenizer_revision,model_path=a.model_path,tokenizer_path=a.tokenizer_path,device=a.device,temperature=a.temperature,top_p=a.top_p,sample_count=a.sample_count,seed=a.seed)
    try:m=run_adjusted_evaluation(mode=a.mode,raw_dir=a.raw_dir,source_manifest=a.source_manifest,factor_manifest=a.factor_manifest,universe_path=a.universe,output_dir=a.output,config=cfg,symbols=a.symbols,shard_index=a.shard_index,strict=a.strict,adapter=adapter)
    except ValueError as exc:raise SystemExit(str(exc))
    print(json.dumps(m,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())

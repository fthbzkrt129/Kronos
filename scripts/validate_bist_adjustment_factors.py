#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,sys
from pathlib import Path
import pandas as pd
if __package__ in {None,""}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bist_data.universe import load_universe
from bist_eval.adjustments import aggregate_factor_fingerprint,build_factor_diagnostics,factor_manifest_record
from bist_eval.data import discover_raw_symbol_files,load_raw_symbol_frame
def _write(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(text,encoding="utf-8");os.replace(tmp,path)
def _parse(values):
    if not values:return None
    out=[]
    for v in values:out.extend(x.strip().upper() for x in v.split(",") if x.strip())
    return list(dict.fromkeys(out))
def run_factor_validation(*,raw_dir,source_manifest,universe_path,output_dir,tolerance=1e-8,symbols=None,strict=False,source_artifact_id=None,source_artifact_digest=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);(out/"COMPLETED").unlink(missing_ok=True)
    ordered=[e.symbol for e in load_universe(universe_path)];requested=_parse(symbols);selected=ordered if requested is None else requested;unknown=sorted(set(selected)-set(ordered))
    if unknown:raise ValueError("unknown symbols: "+", ".join(unknown))
    files=discover_raw_symbol_files(raw_dir,selected);diagnostics=[];failures=[]
    for symbol in selected:
        try:
            if symbol not in files:raise ValueError("raw symbol file missing")
            frame=load_raw_symbol_frame(files[symbol]);diagnostics.append(build_factor_diagnostics(symbol,frame,tolerance))
        except Exception as exc:failures.append({"symbol":symbol,"error_type":type(exc).__name__,"error":str(exc)})
    if strict and failures:raise ValueError("factor validation failed: "+", ".join(x["symbol"] for x in failures))
    records=[factor_manifest_record(d) for d in diagnostics];aggregate=aggregate_factor_fingerprint(diagnostics)
    source_manifest_path=Path(source_manifest) if source_manifest else None
    payload={"schema_version":1,"formula_version":"origin-rebased-v1","material_factor_tolerance":tolerance,"source_artifact_id":source_artifact_id,"source_artifact_digest":source_artifact_digest,"source_manifest_digest":hashlib.sha256(source_manifest_path.read_bytes()).hexdigest() if source_manifest_path and source_manifest_path.is_file() else None,"universe_digest":hashlib.sha256(Path(universe_path).read_bytes()).hexdigest(),"summary":{"requested":len(selected),"succeeded":len(diagnostics),"failed":len(failures)},"aggregate_factor_fingerprint":aggregate,"symbols":records,"failures":failures,"static_adjusted_model_inputs_written":False}
    pd.DataFrame(records).to_csv(out/"factor_diagnostics.csv",index=False);_write(out/"factor_manifest.json",json.dumps(payload,ensure_ascii=False,indent=2)+"\n");_write(out/"COMPLETED","ok\n");return payload
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--raw-dir",type=Path,required=True);p.add_argument("--source-manifest",type=Path);p.add_argument("--universe",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--material-factor-tolerance",type=float,default=1e-8);p.add_argument("--symbols",nargs="+");p.add_argument("--strict",action="store_true");p.add_argument("--source-artifact-id");p.add_argument("--source-artifact-digest")
    a=p.parse_args(argv)
    try:r=run_factor_validation(raw_dir=a.raw_dir,source_manifest=a.source_manifest,universe_path=a.universe,output_dir=a.output,tolerance=a.material_factor_tolerance,symbols=a.symbols,strict=a.strict,source_artifact_id=a.source_artifact_id,source_artifact_digest=a.source_artifact_digest)
    except ValueError as exc:raise SystemExit(str(exc))
    print(json.dumps(r,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())

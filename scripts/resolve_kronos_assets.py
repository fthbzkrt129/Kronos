#!/usr/bin/env python3
"""Resolve and snapshot exact public Kronos model assets."""
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
if __package__ in {None,""}: sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def resolve_assets(*,model_id,tokenizer_id,output_dir,api=None,snapshot=None):
    if api is None:
        from huggingface_hub import HfApi
        api=HfApi()
    if snapshot is None:
        from huggingface_hub import snapshot_download
        snapshot=snapshot_download
    root=Path(output_dir); root.mkdir(parents=True,exist_ok=True); model_sha=api.model_info(model_id).sha; tokenizer_sha=api.model_info(tokenizer_id).sha
    model_dir=root/"model"; tokenizer_dir=root/"tokenizer"; snapshot(repo_id=model_id,revision=model_sha,local_dir=str(model_dir)); snapshot(repo_id=tokenizer_id,revision=tokenizer_sha,local_dir=str(tokenizer_dir))
    manifest={"schema_version":1,"generated_at":datetime.now(timezone.utc).isoformat(),"model_id":model_id,"model_revision":model_sha,"model_path":str(model_dir),"tokenizer_id":tokenizer_id,"tokenizer_revision":tokenizer_sha,"tokenizer_path":str(tokenizer_dir)}
    (root/"asset_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8"); return manifest
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--model-id",default="NeoQuasar/Kronos-mini"); p.add_argument("--tokenizer-id",default="NeoQuasar/Kronos-Tokenizer-2k"); p.add_argument("--output",type=Path,required=True); a=p.parse_args(argv)
    print(json.dumps(resolve_assets(model_id=a.model_id,tokenizer_id=a.tokenizer_id,output_dir=a.output),indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())

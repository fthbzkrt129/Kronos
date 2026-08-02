#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from datetime import datetime,timezone
from pathlib import Path
if __package__ in {None,""}:sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
def resolve_assets(*,model_id,tokenizer_id,output_dir,api=None,snapshot=None):
    if api is None:
        from huggingface_hub import HfApi;api=HfApi()
    if snapshot is None:
        from huggingface_hub import snapshot_download;snapshot=snapshot_download
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True);ms=api.model_info(model_id).sha;ts=api.model_info(tokenizer_id).sha;md=root/"model";td=root/"tokenizer";snapshot(repo_id=model_id,revision=ms,local_dir=str(md));snapshot(repo_id=tokenizer_id,revision=ts,local_dir=str(td));m={"schema_version":1,"generated_at":datetime.now(timezone.utc).isoformat(),"model_id":model_id,"model_revision":ms,"model_path":str(md),"tokenizer_id":tokenizer_id,"tokenizer_revision":ts,"tokenizer_path":str(td)};(root/"asset_manifest.json").write_text(json.dumps(m,indent=2)+"\n");return m
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--model-id",required=True);p.add_argument("--tokenizer-id",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args(argv);print(json.dumps(resolve_assets(model_id=a.model_id,tokenizer_id=a.tokenizer_id,output_dir=a.output),indent=2));return 0
if __name__=="__main__":raise SystemExit(main())

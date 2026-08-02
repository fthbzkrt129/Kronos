import json
from pathlib import Path
import pandas as pd
from bist_eval.config import EvaluationConfig
from scripts.evaluate_bist100_zero_shot import run_evaluation
from scripts.reduce_bist100_zero_shot import run_reducer
from scripts.resolve_kronos_assets import resolve_assets
class FakePredictor:
    def predict_batch(self,dfs,xs,ys,**kwargs): return [pd.DataFrame({"close":[float(df.close.iloc[-1])]*len(y)},index=pd.DatetimeIndex(y)) for df,y in zip(dfs,ys)]
class FakeAdapter:
    def __init__(self): self.predictor=FakePredictor(); self.calls=0
    def predict_cohort(self,windows): self.calls+=1; return {w.symbol:self.predictor.predict_batch([w.context],[w.context.timestamps],[pd.Series(w.target_timestamps)],pred_len=5)[0].close.to_numpy() for w in windows}
def setup_data(tmp_path,frame_factory,symbols=("AAA","BBB")):
    data=tmp_path/"data"; data.mkdir(); universe=tmp_path/"universe.csv"; universe.write_text("symbol,name,valid_from,valid_to\n"+"".join(f"{s},{s},2026-07-01,2026-09-30\n" for s in symbols))
    for i,s in enumerate(symbols): frame_factory(950,start="2022-01-03",close_start=100+i).to_csv(data/f"{s}.csv",index=False)
    return data,universe
def test_evaluate_subset_writes_valid_artifact(tmp_path,frame_factory):
    data,u=setup_data(tmp_path,frame_factory); out=tmp_path/"out"; cfg=EvaluationConfig(start_date="2024-01-01",end_date="2025-01-31",lookback=400,horizon=5,calendar_coverage=1,shard_count=2,minimum_ranking_cohort=1)
    manifest=run_evaluation(data_dir=data,universe_path=u,output_dir=out,config=cfg,symbols=["AAA"],strict=True,model_adapter=FakeAdapter())
    assert manifest["symbols"]==["AAA"] and (out/"COMPLETED").is_file() and len(pd.read_csv(out/"predictions.csv"))>0
def make_shards(tmp_path):
    root=tmp_path/"shards"; root.mkdir()
    for i in range(2):
        d=root/f"shard-{i}"; d.mkdir(); (d/"COMPLETED").write_text("ok")
        manifest={"shard_index":i,"shard_count":2,"symbols":[f"S{i}"],"config_fingerprint":"c","source_data_fingerprint":"d","model_revision":"m","tokenizer_revision":"t"}; (d/"shard_manifest.json").write_text(json.dumps(manifest))
        pred=[]
        for method in ["kronos","last_close"]:
            for h in range(1,6): pred.append({"symbol":f"S{i}","candidate_month":"2026-01","forecast_origin":"2026-01-02","target_timestamp":f"2026-01-{h+2:02}","horizon_step":h,"method":method,"predicted_close":10+h,"actual_close":10+h,"history_last_close":10})
        pd.DataFrame(pred).to_csv(d/"predictions.csv",index=False)
        from bist_eval.metrics import compute_window_metrics
        compute_window_metrics(pd.DataFrame(pred).assign(forecast_origin=pd.Timestamp("2026-01-02"))).to_csv(d/"window_metrics.csv",index=False)
        pd.DataFrame(columns=["symbol","candidate_month","reason_code","reason_detail","available_history_rows","available_target_rows"]).to_csv(d/"skipped_windows.csv",index=False)
    return root
def test_reducer_combines_shards(tmp_path):
    root=make_shards(tmp_path); out=tmp_path/"final"; summary=run_reducer(shards_dir=root,expected_shards=2,output_dir=out,minimum_ranking_cohort=1); assert summary["symbols_evaluated"]==2 and (out/"COMPLETED").is_file()
class Info:
    def __init__(self,sha): self.sha=sha
class API:
    def model_info(self,id): return Info("sha-"+id.split("/")[-1])
def test_asset_resolver_records_exact_revisions(tmp_path):
    calls=[]
    def snap(**kwargs): calls.append(kwargs); Path(kwargs["local_dir"]).mkdir(parents=True,exist_ok=True); return kwargs["local_dir"]
    m=resolve_assets(model_id="a/m",tokenizer_id="a/t",output_dir=tmp_path,api=API(),snapshot=snap); assert m["model_revision"]=="sha-m" and len(calls)==2
    payload=json.dumps(m).lower(); assert "access_token" not in payload and "hf_token" not in payload

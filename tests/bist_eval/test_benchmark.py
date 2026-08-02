import numpy as np
import pandas as pd
from bist_eval.benchmark import run_mini_pair_shard,run_small_shard
from bist_eval.calendar import MonthlyCohort
from bist_eval.config import AdjustedBenchmarkConfig

class Adapter:
    def __init__(self):self.contexts=[]
    def predict_cohort(self,windows):self.contexts.extend([w.context.copy() for w in windows]);return {w.symbol:np.repeat(float(w.context.close.iloc[-1]),len(w.target_timestamps)) for w in windows}

def fixture():
    rows=410;ts=pd.bdate_range("2022-01-03",periods=rows);close=100+np.arange(rows)*.1;f=np.ones(rows);f[:200]=.5
    raw=pd.DataFrame({"timestamps":ts,"open":close-.2,"high":close+.5,"low":close-.5,"close":close,"adj_close":close*f,"volume":1000+np.arange(rows),"symbol":"AAA","yahoo_symbol":"AAA.IS"});origin=ts[404];return {"AAA":raw},[MonthlyCohort("2023-07",origin,tuple(ts[405:410]))]

def test_mini_pair_uses_common_target_and_separate_contexts():
    frames,cohorts=fixture();adapter=Adapter();result=run_mini_pair_shard(raw_frames=frames,cohorts=cohorts,symbols=["AAA"],config=AdjustedBenchmarkConfig(),adapter=adapter);assert set(result.predictions.experiment_arm)=={"raw-mini","adjusted-mini","adjusted-baselines"};assert result.predictions.groupby(["symbol","target_timestamp"]).actual_close.nunique().max()==1;assert len(adapter.contexts)==2 and not adapter.contexts[0].equals(adapter.contexts[1])

def test_small_arm_does_not_duplicate_baselines():
    frames,cohorts=fixture();config=AdjustedBenchmarkConfig(experiment_arm="adjusted-small",model_id="NeoQuasar/Kronos-small",tokenizer_id="NeoQuasar/Kronos-Tokenizer-base");result=run_small_shard(raw_frames=frames,cohorts=cohorts,symbols=["AAA"],config=config,adapter=Adapter());assert set(result.predictions.experiment_arm)=={"adjusted-small"} and set(result.predictions.method)=={"kronos"}

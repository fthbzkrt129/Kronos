import pandas as pd,pytest
from bist_eval.model_adapter import KronosModelAdapter,derive_cohort_seed
from bist_eval.windows import ForecastWindow
class FakePredictor:
    def __init__(self,bad=False): self.calls=[]; self.bad=bad
    def predict_batch(self,dfs,xs,ys,**kwargs):
        self.calls.append((dfs,xs,ys,kwargs)); out=[]
        for y in ys:
            n=len(y)-(1 if self.bad else 0); out.append(pd.DataFrame({"close":[10.0]*n},index=pd.DatetimeIndex(y.iloc[:n])))
        return out
def make_window(frame_factory,symbol="X"):
    f=frame_factory(405); return ForecastWindow(symbol,"2024-07",f.timestamps.iloc[399],tuple(f.timestamps.iloc[400:405]),f.iloc[:400].copy(),f.iloc[400:405].copy())
def test_predict_cohort_contract(frame_factory):
    fake=FakePredictor(); adapter=KronosModelAdapter(predictor=fake,seed=7); w=make_window(frame_factory); out=adapter.predict_cohort([w])
    assert list(out)==["X"] and len(out["X"])==5; dfs,xs,ys,kw=fake.calls[0]; assert list(dfs[0].columns)==["open","high","low","close","volume","amount"] and len(xs[0])==400 and kw["verbose"] is False
def test_invalid_output_fails(frame_factory):
    with pytest.raises(ValueError,match="schema"): KronosModelAdapter(predictor=FakePredictor(True)).predict_cohort([make_window(frame_factory)])
def test_seed_is_stable(): assert derive_cohort_seed(1,pd.Timestamp("2026-01-01"))==derive_cohort_seed(1,pd.Timestamp("2026-01-01"))

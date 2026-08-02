import pandas as pd,pytest
from bist_eval.calendar import MonthlyCohort
from bist_eval.data import load_symbol_frame,load_timestamp_coverage
from bist_eval.windows import build_symbol_windows
def test_load_symbol_frame_sorts_and_preserves_amount(tmp_path,frame_factory):
    f=frame_factory(10).iloc[::-1]; p=tmp_path/"X.csv"; f.to_csv(p,index=False)
    out=load_symbol_frame(p); assert out.timestamps.is_monotonic_increasing; assert list(out.columns)==["timestamps","open","high","low","close","volume","amount"]
    assert out.iloc[0].amount==pytest.approx(frame_factory(10).iloc[0].amount)
def test_loader_rejects_missing_amount(tmp_path,frame_factory):
    p=tmp_path/"X.csv"; frame_factory(3).drop(columns="amount").to_csv(p,index=False)
    with pytest.raises(ValueError,match="amount"): load_symbol_frame(p)
def test_load_timestamp_coverage_reads_dates(tmp_path,frame_factory):
    p=tmp_path/"X.csv"; frame_factory(3).to_csv(p,index=False); out=load_timestamp_coverage({"X":p}); assert len(out["X"])==3
def test_exact_leakage_free_window(frame_factory):
    frame=frame_factory(410); origin=frame.timestamps.iloc[404]; targets=tuple(frame.timestamps.iloc[405:410]); cohort=MonthlyCohort(origin.strftime("%Y-%m"),origin,targets)
    windows,skips=build_symbol_windows("X",frame,[cohort],lookback=400,horizon=5); w=windows[0]
    assert not skips and len(w.context)==400 and len(w.target)==5 and w.context.timestamps.max()==origin and w.context.timestamps.max()<w.target.timestamps.min()
def test_recent_ipo_is_skipped(frame_factory):
    frame=frame_factory(100); origin=frame.timestamps.iloc[94]; cohort=MonthlyCohort("2023-05",origin,tuple(frame.timestamps.iloc[95:100])); w,s=build_symbol_windows("X",frame,[cohort],lookback=400,horizon=5)
    assert not w and s[0].reason_code=="insufficient_history"
def test_missing_target_is_skipped(frame_factory):
    frame=frame_factory(410); origin=frame.timestamps.iloc[404]; cohort=MonthlyCohort("2024-01",origin,(frame.timestamps.iloc[405],pd.Timestamp("2099-01-01"),*tuple(frame.timestamps.iloc[407:410])))
    w,s=build_symbol_windows("X",frame,[cohort],lookback=400,horizon=5); assert not w and s[0].reason_code=="missing_target_date"

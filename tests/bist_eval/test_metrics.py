import pandas as pd,pytest
from bist_eval.metrics import compute_window_metrics,aggregate_symbol_metrics,compute_ranking_metrics,kronos_win_rates
def predictions(n=25):
    rows=[]
    for i in range(n):
        for method,offset in [("kronos",.1),("last_close",1.0)]:
            for h in range(1,6): rows.append({"symbol":f"S{i:02}","candidate_month":"2026-01","forecast_origin":pd.Timestamp("2026-01-02"),"target_timestamp":pd.Timestamp("2026-01-02")+pd.Timedelta(days=h),"horizon_step":h,"method":method,"predicted_close":100+i+h+offset,"actual_close":100+i+h,"history_last_close":100+i})
    return pd.DataFrame(rows)
def test_window_metrics_and_win_rate():
    wm=compute_window_metrics(predictions()); k=wm[wm.method=="kronos"].iloc[0]; assert k.mae==pytest.approx(.1) and k.direction_correct
    assert kronos_win_rates(wm)[0]["kronos_win_rate"]==1.0
def test_symbol_and_ranking_metrics():
    wm=compute_window_metrics(predictions()); sm=aggregate_symbol_metrics(wm); rm=compute_ranking_metrics(wm,20)
    assert len(sm)==50 and rm.ranking_available.all() and rm.top5_overlap.min()==5
def test_zero_actual_final_ape_is_nan():
    p=predictions(1); p.loc[p.horizon_step==5,"actual_close"]=0; wm=compute_window_metrics(p); assert wm.final_ape.isna().all()

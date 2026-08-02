import json,pandas as pd,pytest
from bist_eval.reporting import write_shard_output,write_reduced_output
def frames():
    pred=pd.DataFrame([{"symbol":"X","candidate_month":"2026-01","forecast_origin":pd.Timestamp("2026-01-02"),"target_timestamp":pd.Timestamp("2026-01-05"),"horizon_step":1,"method":"kronos","predicted_close":1.0,"actual_close":1.1,"history_last_close":.9}])
    wm=pd.DataFrame([{"symbol":"X","candidate_month":"2026-01","forecast_origin":pd.Timestamp("2026-01-02"),"method":"kronos","mae":.1,"rmse":.1,"final_ape":.09,"final_abs_error":.1,"predicted_return_5d":.1,"actual_return_5d":.2,"direction_correct":True}])
    sk=pd.DataFrame(columns=["symbol","candidate_month","reason_code","reason_detail","available_history_rows","available_target_rows"]); return pred,wm,sk
def test_shard_completion_written_last(tmp_path):
    p,w,s=frames(); write_shard_output(tmp_path,p,w,s,{"shard_index":0}); assert (tmp_path/"COMPLETED").is_file() and json.loads((tmp_path/"shard_manifest.json").read_text())["shard_index"]==0
def test_duplicate_prediction_rejected(tmp_path):
    p,w,s=frames()
    with pytest.raises(ValueError,match="duplicate"): write_shard_output(tmp_path,pd.concat([p,p]),w,s,{})
def test_reduced_report_contains_warnings(tmp_path):
    p,w,s=frames(); write_reduced_output(tmp_path,predictions=p,window_metrics=w,skips=s,symbol_metrics=pd.DataFrame(),period_metrics=pd.DataFrame(),ranking_metrics=pd.DataFrame(),summary={"eligible_windows":1},manifest={})
    text=(tmp_path/"report.md").read_text(); assert "survivorship" in text and "not investment advice" in text

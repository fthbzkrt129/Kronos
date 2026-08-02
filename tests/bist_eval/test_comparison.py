import pandas as pd
import pytest
from bist_eval.comparison import ComparisonSpec,build_paired_comparison,clustered_bootstrap_mean_difference,summarize_comparison_intervals

def metrics(differences=(-.1,-.2,-.3,-.4)):
    rows=[]
    for i,difference in enumerate(differences):
        for arm,error in (("adjusted-mini",1+difference),("raw-mini",1)):
            rows.append({"experiment_arm":arm,"symbol":f"S{i%2}","candidate_month":f"2026-{i+1:02}","forecast_origin":pd.Timestamp("2026-01-01")+pd.offsets.MonthBegin(i),"method":"kronos","log_return_abs_error":error,"final_target_timestamp":pd.Timestamp("2026-01-10")+pd.offsets.MonthBegin(i),"history_last_close":100.,"actual_final_close":101.,"scoring_target_view":"origin_rebased","exposure_bucket":"no_material_change","common_target_fingerprint":"x"})
    return pd.DataFrame(rows)

def test_paired_difference_orientation_and_target_validation():
    spec=ComparisonSpec("x","adjusted-mini","kronos","raw-mini","kronos");paired=build_paired_comparison(metrics(),spec);assert (paired.difference<0).all() and set(paired.winner)=={"challenger"};bad=metrics();bad.loc[(bad.experiment_arm=="raw-mini")&(bad.symbol=="S0"),"actual_final_close"]=999
    with pytest.raises(ValueError,match="target mismatch"):build_paired_comparison(bad,spec)

def test_clustered_bootstrap_is_deterministic_and_robust():
    paired=build_paired_comparison(metrics(),ComparisonSpec("x","adjusted-mini","kronos","raw-mini","kronos"));a=summarize_comparison_intervals(paired,draws=500,confidence=.95,seed=7);b=summarize_comparison_intervals(paired,draws=500,confidence=.95,seed=7);pd.testing.assert_frame_equal(a,b);assert set(a.decision)=={"robustly_better"}

def test_single_cluster_is_unavailable():
    paired=pd.DataFrame({"symbol":["A","A"],"forecast_origin":[1,2],"difference":[-.1,-.2]});result=clustered_bootstrap_mean_difference(paired,cluster_column="symbol",draws=20,confidence=.95,seed=1);assert result["available"] is False

import pandas as pd
from bist_eval.calendar import build_canonical_calendar,build_monthly_cohorts
def test_calendar_ceil_coverage_and_deterministic():
    a=pd.DatetimeIndex(["2026-01-02","2026-01-05","2026-01-06"]); b=pd.DatetimeIndex(["2026-01-02","2026-01-05"]); c=pd.DatetimeIndex(["2026-01-02","2026-01-06"])
    left=build_canonical_calendar({"a":a,"b":b,"c":c},coverage_threshold=.67,start_date="2026-01-01",end_date="2026-01-31")
    right=build_canonical_calendar({"c":c,"a":a,"b":b},coverage_threshold=.67,start_date="2026-01-01",end_date="2026-01-31")
    assert list(left)==[pd.Timestamp("2026-01-02")] and left.equals(right)
def test_monthly_cohorts_share_next_five_dates():
    cal=pd.bdate_range("2026-01-01",periods=30); cohorts,skips=build_monthly_cohorts(cal,horizon=5)
    assert cohorts[0].forecast_origin==pd.Timestamp("2026-01-01") and cohorts[0].target_timestamps==tuple(cal[1:6])
    assert skips==[] or skips[-1].reason_code=="incomplete_target_calendar"
def test_incomplete_final_month_is_skipped():
    cal=pd.DatetimeIndex(["2026-01-30","2026-02-02","2026-02-03"]); cohorts,skips=build_monthly_cohorts(cal,horizon=5)
    assert not cohorts and len(skips)==2

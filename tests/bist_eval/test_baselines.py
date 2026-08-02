import numpy as np,pandas as pd,pytest
from bist_eval.baselines import forecast_baselines
def test_baseline_formulas():
    close=np.arange(1,21,dtype=float); out=forecast_baselines(pd.DataFrame({"close":close}),5)
    assert np.all(out["last_close"]==20); assert out["linear_trend_20"][0]==pytest.approx(21)
    rate=(20/1)**(1/19)-1; assert out["momentum_20"][0]==pytest.approx(20*(1+rate))
def test_constant_series():
    out=forecast_baselines(pd.DataFrame({"close":[5.0]*20}),3); assert all(np.allclose(v,5) for v in out.values())
def test_nonpositive_momentum_rejected():
    with pytest.raises(ValueError,match="positive"): forecast_baselines(pd.DataFrame({"close":[0.0]+[1.0]*19}),5)

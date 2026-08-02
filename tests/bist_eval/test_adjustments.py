import numpy as np
import pandas as pd
import pytest
from bist_eval.adjustments import build_factor_diagnostics,classify_exposure,rebase_context,transform_target_after_prediction,validate_provider_factors
from bist_eval.data import load_raw_symbol_frame

def raw_frame(rows=6,factors=None,symbol="AAA"):
    ts=pd.bdate_range("2024-01-02",periods=rows);close=100+np.arange(rows,dtype=float);f=np.ones(rows) if factors is None else np.asarray(factors,float)
    return pd.DataFrame({"timestamps":ts,"open":close-.2,"high":close+.5,"low":close-.5,"close":close,"adj_close":close*f,"volume":1000+np.arange(rows),"symbol":symbol,"yahoo_symbol":symbol+".IS"})

def test_provider_factor_and_fingerprint_are_exact_and_stable():
    raw=load_raw_symbol_frame(raw_frame(3,[.5,.5,1]));np.testing.assert_allclose(validate_provider_factors(raw),[.5,.5,1]);a=build_factor_diagnostics("AAA",raw,1e-8);b=build_factor_diagnostics("AAA",raw,1e-8);assert a.factor_fingerprint==b.factor_fingerprint

def test_origin_rebase_preserves_origin_and_volume():
    raw=raw_frame(3,[.5,.5,1]);out,_=rebase_context(raw,validate_provider_factors(raw),1);assert out.close.iloc[-1]==pytest.approx(raw.close.iloc[-1]);np.testing.assert_allclose(out.volume,raw.volume)

def test_common_factor_multiplier_cancels():
    raw=raw_frame(3,[.5,.75,1]);f=validate_provider_factors(raw);a,_=rebase_context(raw,f,1);b,_=rebase_context(raw,f*4,4);pd.testing.assert_frame_equal(a,b)

def test_target_transform_is_post_prediction_and_exposure_is_recorded():
    raw=raw_frame(5,[1,1,1,.5,.5]);context,_=rebase_context(raw.iloc[:3],np.ones(3),1);target=transform_target_after_prediction(raw.iloc[3:].reset_index(drop=True),np.array([.5,.5]),1);exposure=classify_exposure(np.ones(3),np.array([.5,.5]),1,1e-8);assert context.close.iloc[-1]==raw.close.iloc[2];assert target.close.iloc[0]==pytest.approx(raw.close.iloc[3]*.5);assert exposure.exposure_bucket=="material_factor_change"

def test_invalid_raw_factor_inputs_fail_closed():
    raw=raw_frame(3);raw.loc[1,"adj_close"]=0
    with pytest.raises(ValueError):load_raw_symbol_frame(raw)

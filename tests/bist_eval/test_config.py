import pytest
from bist_eval.config import EvaluationConfig
def test_defaults_and_stable_fingerprint():
    a=EvaluationConfig(); b=EvaluationConfig()
    assert a.fingerprint==b.fingerprint and a.lookback==400 and a.horizon==5
    assert a.model_id=="NeoQuasar/Kronos-mini" and a.tokenizer_id=="NeoQuasar/Kronos-Tokenizer-2k"
@pytest.mark.parametrize("kwargs,match",[({"horizon":0},"horizon"),({"calendar_coverage":0},"coverage"),({"top_p":1.1},"top_p"),({"start_date":"x"},"YYYY-MM-DD"),({"start_date":"2026-01-02","end_date":"2026-01-01"},"end_date")])
def test_rejects_invalid(kwargs,match):
    with pytest.raises(ValueError,match=match): EvaluationConfig(**kwargs)
def test_revision_changes_fingerprint(): assert EvaluationConfig().fingerprint!=EvaluationConfig(model_revision="abc").fingerprint

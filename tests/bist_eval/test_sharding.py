import pytest
from bist_eval.sharding import partition_symbols,select_shard,validate_shard_manifests
def manifests(): return [{"shard_index":i,"shard_count":10,"symbols":[f"S{i}"],"config_fingerprint":"c","source_data_fingerprint":"d","model_revision":"m","tokenizer_revision":"t"} for i in range(10)]
def test_partition_is_stable_and_complete():
    symbols=[f"S{i}" for i in range(100)]; parts=partition_symbols(symbols,10); assert all(len(x)==10 for x in parts) and sum((list(x) for x in parts),[])==symbols and select_shard(symbols,10,3)==tuple(symbols[30:40])
def test_manifest_validation(): validate_shard_manifests(manifests(),10)
def test_manifest_mismatch_rejected():
    m=manifests(); m[1]["config_fingerprint"]="x"
    with pytest.raises(ValueError,match="config"): validate_shard_manifests(m,10)

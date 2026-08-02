from __future__ import annotations
def partition_symbols(symbols,shard_count):
    if shard_count<=0:raise ValueError("shard_count must be positive")
    n=len(symbols);base,extra=divmod(n,shard_count);out=[];start=0
    for i in range(shard_count):size=base+(1 if i<extra else 0);out.append(tuple(symbols[start:start+size]));start+=size
    return out
def select_shard(symbols,shard_count,shard_index):
    p=partition_symbols(symbols,shard_count)
    if not 0<=shard_index<len(p):raise ValueError("shard_index out of range")
    return p[shard_index]
def validate_shard_manifests(manifests,expected_count):
    if len(manifests)!=expected_count:raise ValueError("missing shard manifest")
    if sorted(m.get("shard_index") for m in manifests)!=list(range(expected_count)):raise ValueError("missing or duplicate shard index")
    for f in ("config_fingerprint","source_data_fingerprint","model_revision","tokenizer_revision","shard_count"):
        if len({m.get(f) for m in manifests})!=1:raise ValueError(f"mismatched {f}")
    syms=[s for m in manifests for s in m.get("symbols",[])]
    if len(syms)!=len(set(syms)):raise ValueError("overlapping symbols across shards")
def validate_benchmark_shard_manifests(mini_manifests,small_manifests,expected_count):
    if len(mini_manifests)!=expected_count or len(small_manifests)!=expected_count:raise ValueError("missing benchmark shard manifest")
    for group,mode,arms in ((mini_manifests,"mini-pair",{"raw-mini","adjusted-mini","adjusted-baselines"}),(small_manifests,"small",{"adjusted-small"})):
        if sorted(m.get("shard_index") for m in group)!=list(range(expected_count)):raise ValueError("missing or duplicate shard index")
        for m in group:
            if m.get("mode")!=mode:raise ValueError("wrong benchmark mode")
            if set(m.get("experiment_arms",[]))!=arms:raise ValueError("wrong experiment arm set")
        syms=[s for m in group for s in m.get("symbols",[])]
        if len(syms)!=len(set(syms)):raise ValueError("overlapping symbols across benchmark shards")
    shared=("source_data_fingerprint","factor_fingerprint","universe_fingerprint","cohort_fingerprint","common_protocol_fingerprint","common_target_fingerprint","shard_count")
    for f in shared:
        if len({m.get(f) for m in [*mini_manifests,*small_manifests]})!=1:raise ValueError(f"mismatched {f}")
    for i in range(expected_count):
        if mini_manifests[i].get("symbols")!=small_manifests[i].get("symbols"):raise ValueError("shard symbol mapping mismatch")
    if len({(m.get("model_revision"),m.get("tokenizer_revision")) for m in mini_manifests})!=1:raise ValueError("mismatched mini model revisions")
    if len({(m.get("model_revision"),m.get("tokenizer_revision")) for m in small_manifests})!=1:raise ValueError("mismatched small model revisions")

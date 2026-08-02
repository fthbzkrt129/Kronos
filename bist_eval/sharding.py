"""Deterministic shard partitioning and reducer compatibility checks."""
from __future__ import annotations
from collections.abc import Sequence
from typing import Any
def partition_symbols(symbols: Sequence[str], shard_count: int):
    if shard_count<=0: raise ValueError("shard_count must be positive")
    n=len(symbols); base,extra=divmod(n,shard_count); out=[]; start=0
    for i in range(shard_count):
        size=base+(1 if i<extra else 0); out.append(tuple(symbols[start:start+size])); start+=size
    return out
def select_shard(symbols, shard_count, shard_index):
    parts=partition_symbols(symbols,shard_count)
    if not 0<=shard_index<len(parts): raise ValueError("shard_index out of range")
    return parts[shard_index]
def validate_shard_manifests(manifests: Sequence[dict[str,Any]], expected_count:int):
    if len(manifests)!=expected_count: raise ValueError("missing shard manifest")
    indexes=[m.get("shard_index") for m in manifests]
    if sorted(indexes)!=list(range(expected_count)): raise ValueError("missing or duplicate shard index")
    for field in ("config_fingerprint","source_data_fingerprint","model_revision","tokenizer_revision","shard_count"):
        if len({m.get(field) for m in manifests})!=1: raise ValueError(f"mismatched {field}")
    symbols=[]
    for m in manifests: symbols.extend(m.get("symbols",[]))
    if len(symbols)!=len(set(symbols)): raise ValueError("overlapping symbols across shards")

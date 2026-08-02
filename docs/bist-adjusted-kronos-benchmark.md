# BIST 100 Adjusted-Price Kronos Benchmark

This research-only benchmark compares three model arms on identical monthly BIST windows and one common scoring target:

1. raw context with `Kronos-mini` + `Kronos-Tokenizer-2k`;
2. origin-rebased context with the same mini system;
3. the same origin-rebased context with `Kronos-small` + `Kronos-Tokenizer-base`.

It is not a historical point-in-time BIST 100 backtest and does not produce orders or investment recommendations.

## Why static adjusted CSV files are forbidden

Yahoo `adj_close` represents a provider-adjusted series. Writing one static adjusted history can let adjustments known after an old forecast origin influence the scale of that historical context. The benchmark instead validates a provider factor for each row:

```text
provider_factor_t = adj_close_t / close_t
```

For each forecast origin `o`, the 400-row context is rebased in memory:

```text
relative_factor_t(o) = provider_factor_t / provider_factor_o
rebased_price_t = raw_price_t * relative_factor_t(o)
```

At the origin, the relative factor equals one, so raw and rebased origin prices share the same scale. Target factors and target prices are unavailable to the predictor. After prediction completes, held-out target rows are transformed with the same origin-relative formula and used as the common scoring target for all arms.

Volume remains raw. `amount` is estimated from rebased typical price times raw volume.

## Primary metric

The primary paired metric is final five-day log-return absolute error:

```text
abs(log(predicted_final / origin_close) - log(actual_final / origin_close))
```

Lower is better. The reducer also reports price errors, direction accuracy, ranking correlation, top-five diagnostics, and results split by corporate-action exposure.

Paired error differences use:

```text
challenger_error - reference_error
```

Negative favors the challenger. Both symbol-clustered and forecast-origin-clustered bootstrap intervals must remain below zero for `robustly_better`, or above zero for `robustly_worse`. All other outcomes are `mixed_or_inconclusive`.

## Network-free verification

```bash
python -m pip install numpy pandas pytest
python -m pytest tests/bist_eval -v
python -m compileall -q bist_eval \
  scripts/validate_bist_adjustment_factors.py \
  scripts/evaluate_bist100_adjusted_benchmark.py \
  scripts/reduce_bist100_adjusted_benchmark.py
```

These tests use synthetic candles and fake predictors. They do not download Yahoo data, Torch, model weights, or SciPy.

## Factor validation

```bash
python scripts/validate_bist_adjustment_factors.py \
  --raw-dir data/bist/yahoo/raw \
  --source-manifest data/bist/yahoo/manifest.json \
  --universe data/universes/xu100_2026_q3.csv \
  --output data/bist/adjustment-factors \
  --strict
```

The command writes only `factor_manifest.json`, `factor_diagnostics.csv`, and `COMPLETED`. It never writes static adjusted model inputs.

## Real-model workflows

`BIST Adjusted Benchmark Smoke` is manual-only and checks THYAO and ASELS over two monthly origins with both exact model-tokenizer systems. It is a technical gate, not performance evidence.

`BIST 100 Adjusted Kronos Benchmark` is manual-only. It reuses the immutable verified 100-symbol Yahoo artifact, checks its SHA-256, validates all factor series, prepares both models once, runs ten mini-pair shards and ten small shards, and reduces all 20 artifacts.

## Outputs

The final artifact includes predictions, window/symbol/period/ranking metrics, exact paired comparisons, both clustered bootstrap intervals, source and factor provenance, a machine-readable summary, and a Markdown report.

## Limitations

- Yahoo is not an official licensed Borsa Istanbul source.
- The 2026 Q3 constituent snapshot is projected backward and introduces survivorship and selection bias.
- The factor-ratio cancellation behavior is a research assumption requiring independent validation.
- Volume is not reverse-adjusted and amount is estimated.
- Monthly origins do not represent all possible entry dates.
- Better error metrics do not establish profitability.
- No fine-tuning, portfolio construction, transaction costs, broker connection, paper orders, or live orders are included.

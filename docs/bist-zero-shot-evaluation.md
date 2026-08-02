# BIST 100 Zero-Shot Evaluation

This evaluation measures whether the public pretrained `Kronos-mini` model improves five-trading-day close forecasts over three transparent baselines for the companies in the **2026 Q3 BIST 100 snapshot**.

It is not a historical BIST 100 index backtest. Applying the 2026 Q3 constituent list to 2023-2026 creates survivorship and selection bias.

## Protocol

- Common timestamp-only market calendar requiring 80% symbol coverage.
- First canonical trading date of each month is the shared forecast origin.
- The following five canonical dates are the shared target interval.
- Each eligible symbol supplies exactly 400 observed rows ending at the origin close.
- Recent IPOs and missing-date symbols are skipped rather than padded.
- Target prices never enter model context, normalization, baseline construction, or ranking input.

Kronos is compared with:

1. Last close carried forward.
2. Twenty-row compounded momentum.
3. Twenty-row linear close trend.

## Network-free tests

```bash
python -m pip install numpy pandas pytest
python -m pytest tests/bist_eval -v
python -m compileall -q bist_eval \
  scripts/evaluate_bist100_zero_shot.py \
  scripts/reduce_bist100_zero_shot.py \
  scripts/resolve_kronos_assets.py
```

The unit suite uses synthetic candles and fake predictors. It does not call Yahoo or Hugging Face and does not require Torch.

## Local subset smoke

First prepare Yahoo data with the existing downloader and resolve public model assets:

```bash
python scripts/resolve_kronos_assets.py \
  --model-id NeoQuasar/Kronos-mini \
  --tokenizer-id NeoQuasar/Kronos-Tokenizer-2k \
  --output .models/kronos
```

Then evaluate a small subset:

```bash
python scripts/evaluate_bist100_zero_shot.py \
  --data-dir data/bist/yahoo/kronos \
  --source-manifest data/bist/yahoo/manifest.json \
  --universe data/universes/xu100_2026_q3.csv \
  --output results/bist100-zero-shot/smoke \
  --symbols THYAO ASELS \
  --model-path .models/kronos/model \
  --tokenizer-path .models/kronos/tokenizer \
  --strict
```

For reproducible shard fingerprints, also pass the exact model and tokenizer revisions written in `.models/kronos/asset_manifest.json`.

## Full GitHub Actions run

Run **BIST 100 Zero-Shot Evaluation** manually. The first release enforces ten shards. Data and model assets are prepared once, each shard processes a stable contiguous universe slice, and the reducer refuses missing, overlapping, or incompatible shards.

The full workflow is deliberately not scheduled and does not run on every pull request.

## Output

Each shard produces `predictions.csv`, `window_metrics.csv`, `skipped_windows.csv`, `shard_manifest.json`, and `COMPLETED`.

The reducer additionally produces `symbol_metrics.csv`, `period_metrics.csv`, `ranking_metrics.csv`, `summary.json`, `run_manifest.json`, and `report.md`.

The predicted-top-five mean realized return is an **uncosted diagnostic**, not a strategy return. It does not include commissions, spread, slippage, liquidity, risk constraints, or portfolio construction.

## Limitations

- Yahoo is a personal-research source, not an official licensed Borsa Istanbul feed.
- The universe is a current-quarter snapshot projected backward.
- Corporate actions require independent verification.
- `amount` is estimated and is not official turnover.
- Monthly anchors sample only a subset of possible forecast origins.
- Kronos-mini results do not establish performance for other Kronos sizes.
- No broker connection, paper order, live order, or investment recommendation is produced.

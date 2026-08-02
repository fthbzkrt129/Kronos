# BIST 100 Zero-Shot Evaluation Design

## Status

Approved for specification on 2026-08-02. This document defines an evaluation-only milestone. It does not authorize fine-tuning, portfolio deployment, broker integration, or live trading.

## Purpose

Measure whether the pretrained Kronos model provides useful five-trading-day forecasts for the companies in the 2026 Q3 BIST 100 snapshot when evaluated over historical daily data from 2023 through 2026.

The experiment must compare Kronos with simple, transparent baselines and produce reproducible per-window, per-symbol, per-period, and cross-sectional ranking metrics. Results are research evidence, not investment advice.

## Naming and Interpretation

The report must be named and described as:

> Zero-shot historical evaluation of the 2026 Q3 BIST 100 constituent snapshot over 2023-2026.

It must not be described as a historical BIST 100 index backtest. The repository currently contains a 2026 Q3 membership snapshot, not point-in-time constituent histories. Applying the current snapshot backward introduces survivorship and selection bias.

## Scope

### Included

- The 100 symbols in `data/universes/xu100_2026_q3.csv`.
- Daily Kronos-ready CSV files produced by the existing Yahoo research pipeline.
- Evaluation dates from 2023-01-01 through the last available completed daily candle on or before 2026-08-02.
- A 400-trading-day lookback window.
- A five-trading-day forecast horizon.
- One common monthly forecast origin shared by the cross-sectional cohort.
- The pretrained `NeoQuasar/Kronos-mini` model.
- The compatible `NeoQuasar/Kronos-Tokenizer-2k` tokenizer.
- Deterministic baselines and evaluation metrics.
- Reproducible CSV, JSON, and Markdown outputs.
- Unit tests that do not download model weights or call Yahoo.
- A separately triggered integration workflow for the real model evaluation.

### Excluded

- Fine-tuning Kronos or its tokenizer.
- Historical point-in-time BIST 100 reconstruction.
- Intraday forecasting.
- Portfolio optimization, position sizing, risk-factor neutralization, or order simulation.
- Transaction-cost-adjusted strategy claims.
- Broker connectivity, paper orders, or live orders.
- Claims of profitability or production readiness.

## Experiment Population

The experiment starts with all 100 symbols in the 2026 Q3 snapshot. A symbol-window is eligible only when all of the following are true:

1. The source CSV exists and passes the existing candle quality contract.
2. The symbol contains the common monthly forecast-origin date.
3. At least 400 complete daily observations exist up to and including the forecast origin.
4. The symbol contains all five common target dates after the forecast origin.
5. The input and target rows contain finite OHLC values.
6. Dates are strictly increasing and contain no duplicates.

Symbols listed after recent IPOs will naturally contribute fewer evaluation windows. They must not be padded with synthetic history. Skipped windows and their reasons must be counted and reported.

## Common Market Calendar

Cross-sectional ranking metrics are valid only when symbols are compared over the same forecast interval. The evaluator must therefore build one canonical daily calendar before creating symbol windows.

1. Read timestamp coverage from all schema-valid symbol files without reading future prices into any model feature.
2. A date enters the canonical calendar when it is present in at least a configurable percentage of schema-valid symbol files. The default coverage threshold is 80%.
3. For each calendar month, the first canonical date in that month is the forecast origin.
4. The next five canonical dates are the common target dates.
5. A symbol missing the origin or any target date is skipped for that monthly cohort.

Calendar construction uses timestamps only. It must never use returns, prices, eligibility outcomes, or model predictions. The runtime manifest must record the coverage threshold and the symbol-file count used to construct the calendar.

## Forecast Origin and Leakage Prevention

For each symbol and common monthly forecast origin:

1. End the model context at the forecast-origin close.
2. Select exactly the 400 observed symbol rows ending at that origin.
3. Pass the five common target dates after the origin as `y_timestamp`.
4. Load the actual OHLC values for those five dates only after the prediction is produced or through a test harness that keeps targets inaccessible to the predictor and baselines.
5. Never include a target row in normalization, feature construction, baseline input, ranking input, or model context.

The prediction key is `(symbol, forecast_origin, target_timestamp, method)`. All methods in one monthly ranking cohort must share the same forecast origin and five target timestamps.

## Model Configuration

Initial full evaluation configuration:

- Model: `NeoQuasar/Kronos-mini`
- Tokenizer: `NeoQuasar/Kronos-Tokenizer-2k`
- Experiment lookback: 400
- Prediction length: 5
- Temperature: configurable, default `1.0`
- Top-p: configurable, default `0.9`
- Sample count: configurable, default `1` for the first resource-bounded run
- Random seed: required and recorded
- Device: auto-select CUDA when available, otherwise CPU

The runtime manifest must record exact model identifiers, resolved revisions when available, package versions, device, seed, and inference parameters.

## Baselines

Every eligible window must produce the following baseline forecasts using only the 400-row history:

### Last Close

Predict every future close as the final observed close in the lookback window. This is the primary naive forecasting baseline.

### 20-Day Momentum

Calculate the close-to-close return over the final 20 observed history rows. Project that total return across the next five rows using a constant compounded daily rate. The implementation must define and test the exact formula.

### Moving-Average Trend

Fit a simple linear trend to the final 20 closing prices using observation index as the independent variable, then extrapolate five steps. This baseline must remain deterministic and must not use future timestamps or values.

Baselines primarily forecast `close`. Kronos may return full OHLCVA output, but the first milestone evaluates close-price and return performance.

## Metrics

### Per-Window Forecast Metrics

For Kronos and each baseline:

- Five-step close MAE.
- Five-step close RMSE.
- Final-horizon absolute percentage error, with an explicit zero-denominator guard.
- Predicted five-day return measured from the forecast-origin close to the final predicted close.
- Realized five-day return measured from the same origin close to the final actual close.
- Direction correctness based on the sign of predicted versus realized five-day return.

### Per-Symbol Aggregate Metrics

- Window count and skipped-window count.
- Mean and median MAE.
- Mean RMSE.
- Mean final-horizon absolute percentage error.
- Direction accuracy.
- Pearson correlation between predicted and realized five-day returns when statistically defined.
- Kronos win rate versus each baseline by final-horizon absolute error.

### Per-Period Cross-Sectional Metrics

For each common monthly cohort containing enough eligible symbols:

- Spearman rank correlation between predicted and realized five-day returns.
- Top-five overlap: intersection size between the five highest predicted returns and five highest realized returns.
- Mean realized return of the predicted top five, clearly labelled as an uncosted diagnostic rather than a strategy return.
- Eligible symbol count.
- Common forecast origin and target-date range.

A configurable minimum cohort size must be enforced for ranking metrics; default is 20 symbols. Periods below the threshold remain in the output with ranking metrics marked unavailable.

### Overall Summary

- Total requested symbol-months.
- Eligible and skipped windows by reason.
- Number of symbols with at least one eligible window.
- Aggregate metrics for Kronos and all baselines.
- Count and percentage of periods in which Kronos beats each baseline.
- Explicit limitations and non-production disclaimer.

## Architecture

Add an isolated `bist_eval` package without modifying the existing model internals or Yahoo ingestion contract.

### `bist_eval/calendar.py`

Owns timestamp coverage collection, canonical-calendar construction, monthly common forecast origins, common five-date targets, and coverage-threshold diagnostics.

### `bist_eval/windows.py`

Owns validated CSV loading, symbol eligibility checks, leakage-free 400-row context slicing, target isolation, and structured skip reasons.

### `bist_eval/baselines.py`

Owns deterministic last-close, momentum, and moving-average-trend forecasts.

### `bist_eval/model_adapter.py`

Owns lazy Kronos/tokenizer loading, device selection, seed handling, input-column selection, and conversion of predictor output into the evaluation schema. Importing non-model utilities must not download weights.

### `bist_eval/metrics.py`

Owns numerically guarded per-window metrics and aggregate per-symbol/per-period summaries.

### `bist_eval/reporting.py`

Owns stable output schemas, JSON-safe serialization, Markdown report generation, and run-manifest creation.

### `scripts/evaluate_bist100_zero_shot.py`

Composes the modules through a CLI. It accepts input/output paths, date bounds, model identifiers, lookback, horizon, calendar-coverage threshold, batch/chunk controls, seed, sampling parameters, and optional symbol subsets.

## Data Flow

```text
Yahoo/Kronos CSV artifact
        |
        +--> timestamp coverage --> common monthly calendar
        |
        v
validated per-symbol frame
        |
        v
common-origin leakage-free windows
        |
        +--> deterministic baselines
        |
        +--> Kronos-mini inference
        |
        v
per-window predictions and metrics
        |
        +--> per-symbol aggregates
        +--> common-period ranking aggregates
        +--> overall summary
        |
        v
CSV + JSON + Markdown artifact
```

## Execution Strategy

The full run is computationally heavier than the data download. The design must support deterministic chunking by symbol so multiple jobs can run independently and later be reduced.

Recommended initial workflow:

1. A preparation job obtains or regenerates the existing Yahoo data artifact.
2. The preparation job creates and freezes the canonical calendar plus a configuration fingerprint.
3. Ten evaluation shards each receive a stable set of ten symbols and the same frozen calendar.
4. Each shard loads Kronos-mini once and processes its windows sequentially or in safe batches.
5. Every shard uploads predictions, metrics, skips, and its runtime manifest.
6. A reduction job validates shard compatibility, rejects duplicates, combines outputs, computes cross-sectional metrics, and generates the final report.

Sharding must be deterministic from the ordered universe file, not Python hash order. The reducer must fail closed when expected shards are missing, their frozen-calendar hashes differ, or their configuration fingerprints differ.

## Output Contract

Default output directory:

```text
results/bist100-zero-shot/
├── canonical_calendar.csv
├── predictions.csv
├── window_metrics.csv
├── skipped_windows.csv
├── symbol_metrics.csv
├── period_metrics.csv
├── ranking_metrics.csv
├── summary.json
├── run_manifest.json
└── report.md
```

### `canonical_calendar.csv`

At minimum:

- `forecast_origin`
- `target_1`
- `target_2`
- `target_3`
- `target_4`
- `target_5`
- `origin_coverage_count`
- `origin_coverage_ratio`

### `predictions.csv`

At minimum:

- `symbol`
- `forecast_origin`
- `target_timestamp`
- `horizon_step`
- `method`
- `predicted_close`
- `actual_close`
- `history_last_close`
- `predicted_return_5d`
- `actual_return_5d`

### `skipped_windows.csv`

At minimum:

- `symbol`
- `forecast_origin`
- `reason_code`
- `reason_detail`
- `available_history_rows`
- `missing_target_dates`

Output schemas must be versioned in the run manifest.

## Failure Handling

- Missing or malformed symbol files are recorded as symbol-level failures; strict mode fails the preparation job.
- Insufficient history or missing common target dates are normal skips, not exceptions.
- An empty or too-short canonical calendar is fatal.
- Model loading or inference failure is fatal for the affected shard in strict mode.
- NaN or infinite predictions are rejected and recorded; they must not silently enter aggregates.
- Duplicate prediction keys are fatal during reduction.
- Configuration, model-revision, universe, data-manifest, or canonical-calendar mismatch across shards is fatal.
- Existing results are written atomically or into a unique run directory to prevent partial overwrite.
- A completed marker is written only after all expected output files pass schema and row-count validation.

## Testing

### Unit Tests

Network-free tests must cover:

- Canonical-calendar construction from synthetic timestamp coverage.
- Common monthly origin selection.
- Rejection or skipping when a symbol lacks the common origin or target dates.
- Exact 400-row history ending at the origin and five-date target separation.
- No overlap between context and target.
- Recent-IPO insufficient-history skips.
- Last-close baseline.
- Exact 20-day momentum formula.
- Moving-average trend extrapolation.
- MAE, RMSE, return, direction, correlation, and ranking metrics.
- Proof that ranking cohorts contain one shared origin and target-date set.
- Zero denominators and constant-series correlation handling.
- Stable deterministic sharding.
- Reducer rejection of duplicate keys, missing shards, calendar mismatch, and configuration mismatch.
- Output-schema and JSON serialization behavior.

### Model-Adapter Tests

Use a fake predictor to verify:

- Required OHLCVA columns and timestamps passed to Kronos.
- `NeoQuasar/Kronos-mini` and `NeoQuasar/Kronos-Tokenizer-2k` defaults are explicit.
- Seed and inference parameters are propagated.
- Predictions align with the five common held-out timestamps.
- Ground-truth values are not passed to the predictor.
- Invalid output lengths or non-finite values fail closed.

### Integration Tests

A small manually triggered integration job uses two symbols and two common eligible windows with the real Kronos-mini model. The full 100-symbol workflow runs only through manual dispatch or an explicitly approved scheduled workflow, not on every pull request.

## Acceptance Criteria

The milestone is complete when:

1. All network-free tests pass.
2. The real-model smoke evaluation succeeds on at least two symbols and produces schema-valid artifacts.
3. The full sharded evaluation either completes successfully or reports an explicit, reproducible infrastructure limitation.
4. Every prediction can be traced to a symbol, common forecast origin, context interval, common target interval, model configuration, and source-data manifest.
5. No target row is used in its own forecast context or baseline calculation.
6. Every cross-sectional ranking cohort uses exactly one common origin and common five-date target set.
7. The report compares Kronos with all three baselines and includes the survivorship-bias warning.
8. No order-placement or live-trading capability is introduced.

## Security, Cost, and Operational Boundaries

- Workflows use read-only repository permissions unless artifact upload requires the standard Actions channel.
- No brokerage credentials, exchange credentials, or application secrets are required.
- Hugging Face downloads must use public model access unless the model revision later requires authorization; no token is committed.
- Model and dependency caches are treated as performance optimizations, not sources of truth.
- Full workflow concurrency and timeout limits must prevent uncontrolled compute consumption.

## Known Limitations

- Yahoo is a research source, not an official licensed Borsa Istanbul feed.
- The universe is the 2026 Q3 snapshot projected backward, causing survivorship and selection bias.
- Corporate actions depend on the data preparation choices and require independent validation.
- The estimated `amount` field is not official turnover.
- Monthly anchors reduce compute but do not represent all possible forecast origins.
- The 80% canonical-calendar threshold is an experiment rule, not an official exchange calendar.
- A five-day price forecast is not equivalent to a tradable, risk-adjusted strategy.
- Results from Kronos-mini do not establish performance for Kronos-small or Kronos-base.

## Future Milestones

Only after this evaluation is reviewed:

1. Reconstruct point-in-time BIST constituent histories.
2. Add corporate-action and licensed-data validation.
3. Compare Kronos-small and Kronos-base under the same protocol.
4. Add transaction costs and walk-forward portfolio diagnostics.
5. Consider fine-tuning only if zero-shot evidence and data quality justify it.

# BIST Yahoo Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build a tested, research-only pipeline that downloads a versioned BIST 100 universe from Yahoo Finance and converts daily candles into Kronos-compatible CSV files.

**Architecture:** Add an isolated `bist_data` package for universe parsing, Yahoo adaptation, and data-quality validation. Keep the existing Qlib/China fine-tuning path untouched, and compose the new modules through a CLI that writes raw data, normalized data, and a reproducibility manifest.

**Tech Stack:** Python 3.10+, pandas, yfinance, pytest, CSV/JSON.

---

## Chunk 1: Universe and contracts

### Task 1: Add versioned XU100 universe

**Files:**
- Create: `data/universes/xu100_2026_q3.csv`
- Create: `data/universes/xu100_2026_q3.meta.json`
- Create: `bist_data/__init__.py`
- Create: `bist_data/universe.py`
- Test: `tests/bist_data/test_universe.py`

- [x] Write tests for date filtering, `.IS` ticker generation, duplicate symbols, and invalid dates.
- [x] Run `python -m pytest tests/bist_data/test_universe.py -v` and confirm failure.
- [x] Implement `UniverseEntry` and `load_universe`.
- [x] Run the universe tests and confirm pass.
- [x] Commit the versioned universe implementation.

## Chunk 2: Yahoo adapter and quality gates

### Task 2: Normalize and validate Yahoo candles

**Files:**
- Create: `bist_data/yahoo.py`
- Create: `bist_data/quality.py`
- Test: `tests/bist_data/test_yahoo.py`
- Test: `tests/bist_data/test_quality.py`

- [x] Write failing tests for flat/MultiIndex output, retries, empty responses, invalid OHLC, duplicate timestamps, and Kronos amount generation.
- [x] Run targeted tests and confirm failure.
- [x] Implement the minimal adapter and validation functions.
- [x] Run targeted tests and confirm pass.
- [x] Commit the Yahoo adapter and quality gates.

## Chunk 3: CLI and documentation

### Task 3: Add reproducible downloader CLI

**Files:**
- Create: `scripts/download_bist_yahoo.py`
- Create: `docs/bist-data.md`
- Create: `requirements-dev.txt`
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [x] Add `yfinance==1.5.1` to runtime dependencies and pytest to development dependencies.
- [x] Implement CLI arguments, per-symbol output, manifest writing, symbol filtering, and strict failure mode.
- [x] Document installation and example commands.
- [x] Ignore generated BIST data directories.
- [x] Run `python -m pytest tests/bist_data -v`.
- [x] Run `python -m compileall bist_data scripts/download_bist_yahoo.py`.
- [x] Commit the Yahoo download CLI.

## Chunk 4: Verification

### Task 4: Repository verification

- [x] Confirm the universe contains exactly 100 unique symbols.
- [x] Confirm all Yahoo symbols end in `.IS`.
- [x] Confirm no generated market data is committed.
- [x] Run the complete BIST unit-test suite.
- [x] Open a draft pull request describing data-source limitations and the next zero-shot milestone.

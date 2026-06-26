# Calibration Data

This module turns local audited prediction-market data into simulator parameter ranges. It is calibration groundwork, not empirical proof that PM-DFBA reduces stale-quote races.

## What The Audited Data Supports

The local audit of `/Users/darrenmims/Desktop/Prediction Market Data` found substantial trade/fill data, market metadata, and some event/jump labels. These sources can support:

- calibration of trade-size and notional-volume distributions;
- trade-price paths and rolling VWAP paths;
- coarse event-window and jump-window studies;
- rough stale-loss proxies based on trade-price moves;
- market activity and category distributions when metadata is available;
- taker-trade size and notional quantiles from observed fills.

If only trades are used, liquidation exit curves are proxies. They are not displayed-book executable curves because trade prints do not reveal the full resting book available at each price level.

## What The Audited Data Does Not Prove

The audited data does not prove true stale-quote races.

True stale-race measurement requires order add/cancel/modify/fill sequencing, exchange-side timestamps, participant/order identifiers, and enough latency context to know whether a taker execution beat a maker cancellation or repricing update. Trade prints, market metadata, minute bars, and coarse event labels can show price moves and executed trades, but they cannot by themselves establish that a quote was stale or that a latency race occurred.

## Running Calibration

```bash
PYTHONPATH=src python3 -m pm_dfba_sim.run_calibration \
  --data-dir "/Users/darrenmims/Desktop/Prediction Market Data" \
  --out outputs/calibration
```

By default, the runner reads bounded samples from the audited source layout:

- `output/cache/trades_enriched_sample.parquet`
- `output/cache/trades_enriched_sample_v2.parquet`
- `data/polymarket/trump_winner_trades.parquet`
- representative shards from `data/kalshi/trades/*.parquet`
- representative shards from `data/polymarket/trades/*.parquet`
- representative shards from `data/polymarket/legacy_trades/*.parquet`
- `output/cache/market_token_map.parquet`
- `kalshi_audit/data/all_markets.csv`
- representative shards from `data/kalshi/markets/*.parquet`
- representative shards from `data/polymarket/markets/*.parquet`
- `data/polymarket/trump_jumps.parquet`

Large glob expansions are sampled as first, middle, and last files by natural filename sort. Each file is capped by `--max-rows-per-file`, which defaults to `100000`.

You can override inputs with explicit files or glob patterns:

```bash
PYTHONPATH=src python3 -m pm_dfba_sim.run_calibration \
  --data-dir "/Users/darrenmims/Desktop/Prediction Market Data" \
  --trade-input "data/kalshi/trades/trades_0_10000.parquet" \
  --metadata-input "data/kalshi/markets/markets_0_10000.parquet" \
  --event-input "data/polymarket/trump_jumps.parquet" \
  --out outputs/calibration
```

Use `--near-resolution-window` to control terminal-move labeling:

```bash
PYTHONPATH=src python3 -m pm_dfba_sim.run_calibration \
  --data-dir "/Users/darrenmims/Desktop/Prediction Market Data" \
  --out outputs/calibration \
  --near-resolution-window 24h
```

Jumps at or after `close_time - near_resolution_window` are labeled as near-resolution when metadata has a close, expiration, or resolution timestamp. This is a coarse guardrail against contaminating interim jump-size candidates with terminal settlement moves.

## Outputs

The runner writes derived artifacts only. It does not copy raw data.

- `outputs/calibration/market_summary.csv`
- `outputs/calibration/trade_size_summary.csv`
- `outputs/calibration/jump_windows.csv`
- `outputs/calibration/jump_size_distribution.csv`
- `outputs/calibration/interim_jump_size_distribution.csv`
- `outputs/calibration/terminal_jump_size_distribution.csv`
- `outputs/calibration/simulator_parameter_suggestions.json`
- `outputs/calibration/price_paths_sample.png`
- `outputs/calibration/jump_size_distribution.png`
- `outputs/calibration/trade_size_distribution.png`

`jump_windows.csv` keeps one row per `(market_id, timestamp, window)` and records the largest threshold met with `max_threshold_met`. The split jump-size CSVs separate all detected/event-label moves into interim and near-resolution buckets when metadata supports that distinction.

## YES-Equivalent Price Assumptions

The normalizer maps common schemas onto a YES-equivalent probability axis:

- `yes_price` is used directly.
- `no_price` is mapped to `1 - no_price` when no YES price is present.
- Kalshi cent prices are converted to probabilities.
- If only one generic price column exists, it is used as a YES-equivalent proxy and the output records that orientation is unverified.
- If bid/ask fields are present without a trade price, the bid/ask midpoint may be used as a midpoint proxy.
- `final_outcome_price` is not treated as a trade price. Files with only final outcome values are resolution metadata, not trade-path observations.

These assumptions are recorded in normalized rows and should be treated as calibration assumptions, not proof of executable quote state.

The suggestions JSON reports `verified_orientation_trade_count`, `unverified_orientation_trade_count`, `unverified_orientation_share`, `all_row_suggestions`, and `verified_orientation_only_suggestions`. Direction-based fields such as `adverse_jump_probability_proxy` only use verified-orientation interim jumps; when the relevant rows are orientation-unverified, the proxy is `null`.

## Parameter Suggestions

The suggestions file is deliberately conservative:

- `jump_size_interim_candidates` is the safer candidate set for public interim jump simulation.
- `jump_size_unfiltered_candidates` includes all detected/event-label jumps and can be contaminated by terminal moves.
- `terminal_jump_size_candidates` captures near-resolution/final-settlement moves and should not be used as a generic public-jump maximum.
- `taker_trade_size_quantiles` replaces the older liquidation-size wording. Observed trade sizes are not liquidation demand estimates.
- `liquidation_size_status` is `unknown_from_trades_alone` because trades do not reveal forced-exit demand.

Normalized trades are deduplicated by stable trade identifier when one is present, otherwise by `(market_id, timestamp, yes_price, size)`. This avoids double-counting obvious duplicate rows while staying conservative about distinct same-price prints.

These outputs are useful for bounded simulator calibration and paper sensitivity ranges, but they are not safe as direct empirical proof or plug-in simulator truth. Any paper use should preserve the caveats above, especially the separation between interim and terminal moves and the verified-vs-unverified price orientation split.

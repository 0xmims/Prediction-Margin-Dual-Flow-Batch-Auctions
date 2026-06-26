# Calibration Data

This module turns local audited prediction-market data into simulator parameter ranges. It is calibration groundwork, not empirical proof that PM-DFBA reduces stale-quote races.

## What The Audited Data Supports

The local audit of `/Users/darrenmims/Desktop/Prediction Market Data` found substantial trade/fill data, market metadata, and some event/jump labels. These sources can support:

- calibration of trade-size and notional-volume distributions;
- trade-price paths and rolling VWAP paths;
- coarse event-window and jump-window studies;
- rough stale-loss proxies based on trade-price moves;
- market activity and category distributions when metadata is available;
- proxy liquidation-size ranges from observed trade sizes.

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

## Outputs

The runner writes derived artifacts only. It does not copy raw data.

- `outputs/calibration/market_summary.csv`
- `outputs/calibration/trade_size_summary.csv`
- `outputs/calibration/jump_windows.csv`
- `outputs/calibration/jump_size_distribution.csv`
- `outputs/calibration/simulator_parameter_suggestions.json`
- `outputs/calibration/price_paths_sample.png`
- `outputs/calibration/jump_size_distribution.png`
- `outputs/calibration/trade_size_distribution.png`

## YES-Equivalent Price Assumptions

The normalizer maps common schemas onto a YES-equivalent probability axis:

- `yes_price` is used directly.
- `no_price` is mapped to `1 - no_price` when no YES price is present.
- Kalshi cent prices are converted to probabilities.
- If only one generic price column exists, it is used as a YES-equivalent proxy and the output records that orientation is unverified.
- If bid/ask fields are present without a trade price, the bid/ask midpoint may be used as a midpoint proxy.

These assumptions are recorded in normalized rows and should be treated as calibration assumptions, not proof of executable quote state.

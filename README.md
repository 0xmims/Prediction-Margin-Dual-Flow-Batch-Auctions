# PM-DFBA Synthetic Simulator

This repository implements a synthetic simulation scaffold for the research project "Batching the Jump: Dual-Flow Batch Auctions and the Marginability of Prediction Markets."

The core question is whether prediction-market market structure can reduce venue-created stale-quote losses and liquidation shortfalls enough to improve the viability of leverage and cross-margin, holding the event-risk process fixed.

This is a synthetic scaffold, not empirical evidence.

## Marginability

In this repo, a venue design is more marginable if it can achieve one or more of:

- lower bad-debt probability at fixed leverage;
- lower expected shortfall of bad debt or liquidation losses;
- lower required margin at a target bad-debt probability;
- higher maximum safe leverage at a target bad-debt probability;
- comparable maker economics and natural taker execution quality.

The MVP computes bad-debt probability, bad-debt mean, bad-debt expected shortfall, liquidation shortfall, stale quote loss, public stale quote loss, `maker_loss_placeholder`, liquidation trigger rate, effective liquidation depth, safe leverage at a bad-debt tolerance, and a simplified taker delay cost.

## Venue Designs

- `CLOB`: continuous matching abstraction where public news can create stale-quote races when taker latency beats maker cancel latency.
- `FBA`: frequent batch auction with uniform clearing and no time priority inside the batch.
- `DFBA`: dual-flow batch auction with maker/taker segregation and uniform clearing.
- `PM_DFBA`: prediction-market dual-flow batch auction with YES/NO normalization, maker/taker segregation, public-jump volatility-call protection, price-collared auction-only liquidations, backstop liquidity, and side-specific executable exit curves.

The PM-DFBA assumptions are explicit config parameters. The simulator does not hard-code PM-DFBA to win, and terminal instant-resolution jumps can still produce bad debt.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Tests

```bash
pytest
```

## Run Baseline Experiment

```bash
python -m pm_dfba_sim.run_experiment --config configs/baseline.json --out outputs
```

The baseline writes:

- `outputs/trials.csv`
- `outputs/summary.csv`
- `outputs/safe_leverage.csv`
- `outputs/bad_debt_probability.png`
- `outputs/expected_shortfall.png`
- `outputs/stale_loss.png`
- `outputs/safe_leverage.png`

The run also creates `outputs/figures/` and `outputs/tables/` for future richer reporting.

## Run Ablations

```bash
python -m pm_dfba_sim.run_ablations --config configs/baseline.json --out outputs
```

Optional plotting controls:

```bash
python -m pm_dfba_sim.run_ablations --config configs/baseline.json --out outputs --plot-leverage 3
python -m pm_dfba_sim.run_ablations --config configs/baseline.json --out outputs --plot-all-leverages
```

The ablation run compares:

- `CLOB`
- `FBA`
- `DFBA`
- `PM_DFBA_FULL`
- `PM_DFBA_NO_VOL_CALL`
- `PM_DFBA_NO_BACKSTOP_ONLY`
- `PM_DFBA_NO_MAKER_TAKER_SEGREGATION`
- `PM_DFBA_TOXIC_ONLY`
- `PM_DFBA_TOXIC_WITH_BACKSTOP`
- `PM_DFBA_ADVERSE_DEPTH_ONLY`
- `PM_DFBA_ADVERSE_STACK`
- `TERMINAL_JUMP_STRESS`

The ablation runner writes:

- `outputs/ablation_summary.csv`
- `outputs/safe_leverage_by_public_jump_share.csv`
- `outputs/bad_debt_by_backstop_depth.csv`
- `outputs/stale_loss_by_batch_interval.csv`
- `outputs/terminal_jump_stress.csv`
- `outputs/latency_sweep.csv`
- `outputs/safe_leverage_vs_public_jump_share.png`
- `outputs/bad_debt_by_backstop_depth.png`
- `outputs/stale_loss_by_batch_interval.png`
- `outputs/private_information_stress.png`
- `outputs/terminal_jump_failure.png`
- `outputs/latency_race_probability.png`

The sweep values, latency values, collar mode, and toxic-flow stress multipliers live in `configs/baseline.json` so the load-bearing assumptions are visible and reviewable. The default collar mode is `vwap`, which applies the collar to the average primary liquidation fill; `marginal` is stricter and applies the collar to the final executable unit. Null safe-leverage values are left null in the CSV and omitted as zero-valued points in plots.

## Tailscale Replica Feasibility Probe

Probe the read-only Polymarket/Kalshi TimescaleDB replica for a bounded sample
of non-sports markets (top-of-book reconstruction, depth bands, trade/book
alignment, exit-curve proxies, gap checks, resolution joins):

```bash
PREDICTION_DB_HOST=<tailscale-ip> PYTHONPATH=src python3 -m pm_dfba_sim.run_tailscale_probe \
  --out outputs/tailscale_probe \
  --start 2026-05-27T00:00:00Z --end 2026-06-27T00:00:00Z \
  --max-markets 25 --min-volume 2000
```

Credentials come from `PREDICTION_DB_*` environment variables or `~/.pgpass`
and are never written to outputs. This is a feasibility/calibration probe, not
empirical proof; see `docs/tailscale_db_probe.md` for the reconstruction
contract, safety rules, and interpretation guardrails.

## Paper Figures

Generate two deterministic explanatory concept figures for the v0.1 coworker paper draft:

```bash
python3 -m pm_dfba_sim.run_paper_figures --out outputs/paper_figures
```

These PNGs are schematic concept figures, not empirical results. The current simulator does not implement a full PM-DFBA state machine; see `docs/paper_figures.md` for the output paths and intended use.

## MVP Limitations

- The MVP is synthetic and parameterized.
- It does not use real Kalshi or Polymarket data.
- It does not observe real failed cancel/order races.
- It abstracts agent behavior heavily.
- It does not yet model full agent strategies.
- It does not yet include empirical replay.
- Private-information and terminal-resolution sweeps are synthetic stress tests, not empirical estimates.
- It does not prove PM-DFBA is superior; it creates a framework for testing.

## Future Plan

- Add richer structural ablations for protection factors, liquidity depth, latency, public/private information mix, and terminal jump intensity.
- Define real Kalshi and Polymarket data schemas.
- Add empirical replay once message-level or high-quality snapshot/trade data is available.
- Replace the simple liquidation and stale-loss abstractions with richer order-book and auction state machines.

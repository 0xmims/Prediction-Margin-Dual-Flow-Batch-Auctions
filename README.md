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

The MVP computes bad-debt probability, bad-debt mean, bad-debt expected shortfall, liquidation shortfall, stale quote loss, public stale quote loss, maker loss, liquidation trigger rate, effective liquidation depth, safe leverage at a bad-debt tolerance, and a simplified taker delay cost.

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

## MVP Limitations

- The MVP is synthetic and parameterized.
- It does not use real Kalshi or Polymarket data.
- It does not observe real failed cancel/order races.
- It abstracts agent behavior heavily.
- It does not yet model full agent strategies.
- It does not yet include empirical replay.
- It does not yet include private-information sweeps.
- It includes only placeholder terminal-resolution stress support.
- It does not prove PM-DFBA is superior; it creates a framework for testing.

## Future Plan

- Add parameter sweeps and ablations for protection factors, liquidity depth, latency, public/private information mix, and terminal jump intensity.
- Define real Kalshi and Polymarket data schemas.
- Add empirical replay once message-level or high-quality snapshot/trade data is available.
- Replace the simple liquidation and stale-loss abstractions with richer order-book and auction state machines.

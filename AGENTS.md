# Agent Instructions

## Project

This repo implements synthetic and empirical simulations for PM-DFBA and prediction-market marginability.

## Branch Naming

- `codex/<issue-number>-short-name`
- `claude/<issue-number>-short-name`
- `human/<issue-number>-short-name`
- `paper/<issue-number>-short-name`
- `data/<issue-number>-short-name`

## Core Designs

- `CLOB`
- `FBA`
- `DFBA`
- `PM-DFBA`

## Mechanism Constraints

- Do not implement a sealed-bid jump auction in the core MVP.
- Do not make the core MVP depend on jump rebates.
- Liquidations are auction-only, price-collared, and can access backstop liquidity.
- Do not give liquidations blanket priority over natural flow.
- Volatility-call state should protect public jumps, not simply shorten batches mechanically.

## Tests To Run

```bash
pytest
python -m pm_dfba_sim.run_experiment --config configs/baseline.json --out outputs
```

## Coding Standards

- Use Python 3.11+.
- Prefer typed dataclasses where useful.
- Use deterministic seeds.
- Keep assumptions explicit in config.
- Do not hard-code PM-DFBA to win.
- Keep functions small and readable.
- Add tests for math and invariants.

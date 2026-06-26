# Paper Figures

This repository includes a small deterministic generator for explanatory PM-DFBA concept figures intended for the v0.1 coworker paper draft. These figures are schematic illustrations, not empirical results and not evidence from real market data.

Run:

```bash
python3 -m pm_dfba_sim.run_paper_figures --out outputs/paper_figures
```

If running from an uninstalled source checkout, use `PYTHONPATH=src` or install the package with `pip install -e ".[dev]"` first.

The command writes two PNGs to `outputs/paper_figures/`:

- `outputs/paper_figures/clob_vs_pm_dfba_jump_timeline.png`
- `outputs/paper_figures/marginability_episode_trace.png`

`outputs/` is ignored by Git, so generated PNGs are local derived artifacts by default. Regenerate them when drafting or revising the paper rather than committing binary outputs.

## Figure Intent

- `clob_vs_pm_dfba_jump_timeline.png` contrasts a serial CLOB stale-quote race with a PM-DFBA batch around a public interim jump.
- `marginability_episode_trace.png` shows a simplified leveraged YES episode with zero-equity, liquidation-barrier, and executable-exit concepts.

The figures use deterministic synthetic values and do not read audited prediction-market data.

## Simulator Scope Caveat

The current simulator implements simplified venue types (`CLOB`, `FBA`, `DFBA`, and `PM_DFBA`) and parameterized components such as volatility-call protection, terminal-jump stress, backstop-depth sweeps, and latency sweeps.

It does not yet implement a full PM-DFBA state machine or state controller. State-controller graphics should not be used as evidence or as descriptions of the current code path.

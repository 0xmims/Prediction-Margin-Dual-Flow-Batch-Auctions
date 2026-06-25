from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_mpl_cache = Path(tempfile.gettempdir()) / "pm-dfba-matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pm_dfba_sim.metrics import safe_leverage, summarize_trials
from pm_dfba_sim.simulation import load_config, simulate_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic PM-DFBA baseline experiment.")
    parser.add_argument("--config", default="configs/baseline.json", help="Path to a JSON config.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    results = simulate_experiment(config)
    trials = pd.DataFrame([result.to_row() for result in results])
    summary = summarize_trials(results)
    safe = safe_leverage(summary, config)

    trials.to_csv(out_dir / "trials.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    safe.to_csv(out_dir / "safe_leverage.csv", index=False)

    _plot_metric(
        summary,
        metric="bad_debt_probability",
        ylabel="Bad debt probability",
        path=out_dir / "bad_debt_probability.png",
    )
    _plot_metric(
        summary,
        metric="bad_debt_expected_shortfall_99",
        ylabel="Bad debt ES 99",
        path=out_dir / "expected_shortfall.png",
    )
    _plot_metric(
        summary,
        metric="stale_quote_loss_mean",
        ylabel="Mean stale quote loss",
        path=out_dir / "stale_loss.png",
    )
    _plot_safe_leverage(safe, out_dir / "safe_leverage.png")

    print(f"Wrote {len(trials)} trial rows to {out_dir}")


def _plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for venue, group in summary.groupby("venue", sort=True):
        ax.plot(group["leverage"], group[metric], marker="o", label=venue)
    ax.set_xlabel("Leverage")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_safe_leverage(safe: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    values = safe["safe_leverage_at_bad_debt_tolerance"].fillna(0.0)
    ax.bar(safe["venue"], values)
    ax.set_xlabel("Venue")
    ax.set_ylabel("Safe leverage")
    ax.set_title("Safe leverage at bad-debt tolerance")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

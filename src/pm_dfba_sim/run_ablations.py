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

from pm_dfba_sim.ablations import run_ablation_suite
from pm_dfba_sim.simulation import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PM-DFBA ablation and stress experiments.")
    parser.add_argument("--config", default="configs/baseline.json", help="Path to a JSON config.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    config = load_config(args.config)
    outputs = run_ablation_suite(config, out_dir)
    write_ablation_figures(outputs, out_dir, max(config.leverage_values))
    print(f"Wrote ablation outputs to {out_dir}")


def write_ablation_figures(
    outputs: dict[str, pd.DataFrame],
    out_dir: Path,
    plot_leverage: float,
) -> None:
    _plot_safe_leverage_vs_public_share(
        outputs["safe_leverage_by_public_jump_share"],
        out_dir / "safe_leverage_vs_public_jump_share.png",
    )
    _plot_metric_by_x(
        outputs["bad_debt_by_backstop_depth"],
        x_col="backstop_depth_share",
        y_col="bad_debt_probability",
        path=out_dir / "bad_debt_by_backstop_depth.png",
        title="Bad debt by backstop depth",
        ylabel="Bad debt probability",
        leverage=plot_leverage,
    )
    _plot_metric_by_x(
        outputs["stale_loss_by_batch_interval"],
        x_col="batch_interval_ms",
        y_col="stale_plus_delay_cost_mean",
        path=out_dir / "stale_loss_by_batch_interval.png",
        title="Stale plus delay cost by batch interval",
        ylabel="Mean stale plus delay cost",
        leverage=plot_leverage,
    )
    _plot_private_information_stress(
        outputs["private_information_stress"],
        out_dir / "private_information_stress.png",
        plot_leverage,
    )
    _plot_metric_by_x(
        outputs["terminal_jump_stress"],
        x_col="terminal_jump_probability",
        y_col="bad_debt_probability",
        path=out_dir / "terminal_jump_failure.png",
        title="Terminal jump stress",
        ylabel="Bad debt probability",
        leverage=plot_leverage,
    )


def _plot_safe_leverage_vs_public_share(safe: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, group in safe.groupby("mode", sort=True):
        values = pd.to_numeric(
            group["safe_leverage_at_bad_debt_tolerance"], errors="coerce"
        ).fillna(0.0)
        ax.plot(group["public_jump_share"], values, marker="o", label=mode)
    ax.set_xlabel("Public jump share")
    ax.set_ylabel("Safe leverage")
    ax.set_title("Safe leverage vs public jump share")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_metric_by_x(
    summary: pd.DataFrame,
    x_col: str,
    y_col: str,
    path: Path,
    title: str,
    ylabel: str,
    leverage: float,
) -> None:
    plot_df = summary[summary["leverage"] == leverage]
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, group in plot_df.groupby("mode", sort=True):
        ax.plot(group[x_col], group[y_col], marker="o", label=mode)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} at {leverage:g}x")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_private_information_stress(
    summary: pd.DataFrame,
    path: Path,
    leverage: float,
) -> None:
    plot_df = summary[summary["leverage"] == leverage]
    pivot = plot_df.pivot_table(
        index="private_jump_share",
        columns="mode",
        values="stale_quote_loss_mean",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    clob = pivot.get("CLOB")
    for mode in [col for col in pivot.columns if col != "CLOB"]:
        if clob is None:
            values = pivot[mode]
            ylabel = "Mean stale quote loss"
        else:
            values = clob - pivot[mode]
            ylabel = "CLOB stale loss minus scenario stale loss"
        ax.plot(pivot.index, values, marker="o", label=mode)
    ax.set_xlabel("Private/informed jump share")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Private information stress at {leverage:g}x")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()

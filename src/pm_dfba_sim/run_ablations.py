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
    parser.add_argument(
        "--plot-leverage",
        type=float,
        default=None,
        help="Leverage value to use for single-leverage plots. Defaults to the max configured value.",
    )
    parser.add_argument(
        "--plot-all-leverages",
        action="store_true",
        help="Plot all configured leverages where the figure supports leverage filtering.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    config = load_config(args.config)
    outputs = run_ablation_suite(config, out_dir)
    plot_leverage = args.plot_leverage if args.plot_leverage is not None else max(config.leverage_values)
    write_ablation_figures(
        outputs,
        out_dir,
        plot_leverage,
        plot_all_leverages=args.plot_all_leverages,
    )
    print(f"Wrote ablation outputs to {out_dir}")


def write_ablation_figures(
    outputs: dict[str, pd.DataFrame],
    out_dir: Path,
    plot_leverage: float,
    plot_all_leverages: bool = False,
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
        plot_all_leverages=plot_all_leverages,
    )
    _plot_metric_by_x(
        outputs["stale_loss_by_batch_interval"],
        x_col="batch_interval_ms",
        y_col="stale_plus_delay_cost_mean",
        path=out_dir / "stale_loss_by_batch_interval.png",
        title="Stale plus delay cost by batch interval",
        ylabel="Mean stale plus delay cost",
        leverage=plot_leverage,
        plot_all_leverages=plot_all_leverages,
    )
    _plot_private_information_stress(
        outputs["private_information_stress"],
        out_dir / "private_information_stress.png",
        plot_leverage,
        plot_all_leverages,
    )
    _plot_metric_by_x(
        outputs["terminal_jump_stress"],
        x_col="terminal_jump_probability",
        y_col="bad_debt_probability",
        path=out_dir / "terminal_jump_failure.png",
        title="Terminal jump stress",
        ylabel="Bad debt probability",
        leverage=plot_leverage,
        plot_all_leverages=plot_all_leverages,
    )
    _plot_latency_sweep(
        outputs["latency_sweep"],
        out_dir / "latency_race_probability.png",
        plot_leverage,
        plot_all_leverages,
    )


def _plot_safe_leverage_vs_public_share(safe: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for mode, group in safe.groupby("mode", sort=True):
        values = pd.to_numeric(
            group["safe_leverage_at_bad_debt_tolerance"], errors="coerce"
        )
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
    plot_all_leverages: bool = False,
) -> None:
    plot_df, leverage_label = _select_plot_rows(summary, leverage, plot_all_leverages)
    fig, ax = plt.subplots(figsize=(8, 5))
    grouping = ["mode", "leverage"] if plot_all_leverages else ["mode"]
    for keys, group in plot_df.groupby(grouping, sort=True):
        if plot_all_leverages:
            mode, group_leverage = keys
            label = f"{mode} {group_leverage:g}x"
        else:
            label = str(keys[0] if isinstance(keys, tuple) else keys)
        ax.plot(group[x_col], group[y_col], marker="o", label=label)
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} at {leverage_label}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_private_information_stress(
    summary: pd.DataFrame,
    path: Path,
    leverage: float,
    plot_all_leverages: bool = False,
) -> None:
    plot_df, leverage_label = _select_plot_rows(summary, leverage, plot_all_leverages)
    fig, ax = plt.subplots(figsize=(8, 5))
    for group_leverage, leverage_group in plot_df.groupby("leverage", sort=True):
        pivot = leverage_group.pivot_table(
            index="private_jump_share",
            columns="mode",
            values="stale_quote_loss_mean",
            aggfunc="mean",
        )
        clob = pivot.get("CLOB")
        for mode in [col for col in pivot.columns if col != "CLOB"]:
            if clob is None:
                values = pivot[mode]
                ylabel = "Mean stale quote loss"
            else:
                values = clob - pivot[mode]
                ylabel = "CLOB stale loss minus scenario stale loss"
            label = f"{mode} {group_leverage:g}x" if plot_all_leverages else mode
            ax.plot(pivot.index, values, marker="o", label=label)
    ax.set_xlabel("Private/informed jump share")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Private information stress at {leverage_label}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_latency_sweep(
    summary: pd.DataFrame,
    path: Path,
    leverage: float,
    plot_all_leverages: bool = False,
) -> None:
    plot_df, leverage_label = _select_plot_rows(summary, leverage, plot_all_leverages)
    plot_df = plot_df[plot_df["mode"] != "CLOB"]
    fig, ax = plt.subplots(figsize=(8, 5))
    grouping = ["mode", "leverage"] if plot_all_leverages else ["mode"]
    for keys, group in plot_df.groupby(grouping, sort=True):
        if plot_all_leverages:
            mode, group_leverage = keys
            label = f"{mode} {group_leverage:g}x"
        else:
            label = str(keys[0] if isinstance(keys, tuple) else keys)
        averaged = (
            group.groupby("implied_clob_race_probability", as_index=False)[
                "stale_loss_advantage_vs_clob"
            ]
            .mean()
            .sort_values("implied_clob_race_probability")
        )
        ax.plot(
            averaged["implied_clob_race_probability"],
            averaged["stale_loss_advantage_vs_clob"],
            marker="o",
            label=label,
        )
    ax.set_xlabel("Implied CLOB race probability")
    ax.set_ylabel("CLOB stale loss minus scenario stale loss")
    ax.set_title(f"Latency race stress at {leverage_label}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _select_plot_rows(
    summary: pd.DataFrame,
    leverage: float,
    plot_all_leverages: bool,
) -> tuple[pd.DataFrame, str]:
    if plot_all_leverages:
        return summary.copy(), "all leverages"

    leverage_values = pd.to_numeric(summary["leverage"], errors="coerce")
    plot_df = summary[(leverage_values - leverage).abs() < 1e-9]
    if plot_df.empty:
        available = sorted(float(value) for value in leverage_values.dropna().unique())
        raise ValueError(
            f"plot leverage {leverage:g}x is not present in outputs; available values: {available}"
        )
    return plot_df, f"{leverage:g}x"


if __name__ == "__main__":
    main()

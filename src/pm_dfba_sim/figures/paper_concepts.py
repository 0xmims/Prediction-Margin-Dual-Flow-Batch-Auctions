from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

_mpl_cache = Path(tempfile.gettempdir()) / "pm-dfba-matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


PAPER_CONCEPT_FIGURES = (
    "clob_vs_pm_dfba_jump_timeline.png",
    "marginability_episode_trace.png",
    "pm_dfba_state_machine.png",
)

_INK = "#1f2933"
_MUTED = "#667085"
_CLOB = "#b42318"
_PM_DFBA = "#05603a"
_NEWS = "#b54708"
_BATCH = "#d1fadf"
_BARRIER = "#175cd3"
_ZERO_EQUITY = "#7a271a"


def generate_paper_concept_figures(out_dir: str | Path) -> list[Path]:
    """Generate deterministic explanatory PM-DFBA concept figures."""

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    renderers: tuple[tuple[str, Callable[[Path], None]], ...] = (
        (PAPER_CONCEPT_FIGURES[0], plot_clob_vs_pm_dfba_jump_timeline),
        (PAPER_CONCEPT_FIGURES[1], plot_marginability_episode_trace),
        (PAPER_CONCEPT_FIGURES[2], plot_pm_dfba_state_machine),
    )
    written: list[Path] = []
    for filename, renderer in renderers:
        path = output_path / filename
        renderer(path)
        written.append(path)
    return written


def plot_clob_vs_pm_dfba_jump_timeline(path: str | Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    fig.suptitle(
        "Public Interim Jump: Serial CLOB Race vs PM-DFBA Batch",
        fontsize=15,
        fontweight="bold",
        color=_INK,
    )

    _draw_timeline_axis(axes[0], "Panel A: CLOB", _CLOB)
    _draw_event(axes[0], 100, "public news\narrives", _NEWS, y=0.66)
    _draw_event(axes[0], 140, "taker hit\nreaches engine", _CLOB, y=0.30)
    _draw_event(axes[0], 180, "maker cancel\narrives later", _MUTED, y=0.66)
    _draw_event(axes[0], 150, "stale quote\nexecuted", _CLOB, y=0.82)
    _draw_arrow(axes[0], 102, 0.47, 140, 0.47, _CLOB, "latency path")
    _draw_arrow(axes[0], 102, 0.55, 180, 0.55, _MUTED, "cancel path")
    axes[0].text(
        315,
        0.48,
        "serial priority\ncreates race",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=_CLOB,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#fff1f0", edgecolor=_CLOB),
    )

    _draw_timeline_axis(axes[1], "Panel B: PM-DFBA", _PM_DFBA)
    axes[1].add_patch(Rectangle((0, 0.18), 250, 0.58, facecolor=_BATCH, edgecolor=_PM_DFBA, lw=1.8))
    axes[1].text(125, 0.78, "batch window: 0-250ms", ha="center", va="bottom", fontsize=10, color=_PM_DFBA)
    _draw_event(axes[1], 100, "public news\narrives", _NEWS, y=0.66)
    _draw_event(axes[1], 145, "taker order\nsubmitted", _PM_DFBA, y=0.31)
    _draw_event(axes[1], 185, "maker update /\ncancel allowed", _PM_DFBA, y=0.87)
    _draw_event(axes[1], 250, "uniform clearing\nat batch end", _PM_DFBA, y=0.23)
    _draw_arrow(axes[1], 145, 0.42, 250, 0.42, _PM_DFBA, "competes in batch")
    _draw_arrow(axes[1], 185, 0.58, 250, 0.58, _PM_DFBA, "reprices before clearing")
    axes[1].text(
        368,
        0.50,
        "batching converts\nspeed race into\nprice competition",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=_PM_DFBA,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ecfdf3", edgecolor=_PM_DFBA),
    )

    axes[1].set_xlabel("milliseconds from pre-jump reference time", fontsize=11)
    fig.text(
        0.5,
        0.01,
        "Concept figure only: deterministic synthetic timing, not empirical latency evidence.",
        ha="center",
        fontsize=9,
        color=_MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_marginability_episode_trace(path: str | Path) -> None:
    times = [0.00, 0.35, 0.50, 0.75, 1.00]
    prices = [0.60, 0.60, 0.42, 0.42, 0.43]
    zero_equity_price = 0.40
    liquidation_barrier = 0.45
    clob_exit_price = 0.38
    pm_dfba_exit_price = 0.43

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.plot(times, prices, color=_INK, linewidth=2.7, marker="o", label="YES probability path")
    ax.axhspan(0.0, zero_equity_price, color="#fee4e2", alpha=0.7, label="bad-debt region")
    ax.axhline(zero_equity_price, color=_ZERO_EQUITY, linestyle="--", linewidth=2, label="zero-equity price = 0.40")
    ax.axhline(liquidation_barrier, color=_BARRIER, linestyle=":", linewidth=2.5, label="liquidation barrier = 0.45")

    ax.scatter([0.82], [clob_exit_price], s=130, color=_CLOB, zorder=5, label="CLOB exit price = 0.38")
    ax.scatter([0.92], [pm_dfba_exit_price], s=130, color=_PM_DFBA, zorder=5, label="PM-DFBA exit price = 0.43")
    ax.annotate(
        "public jump\n0.60 -> 0.42",
        xy=(0.50, 0.42),
        xytext=(0.38, 0.72),
        arrowprops=dict(arrowstyle="->", lw=1.6, color=_NEWS),
        fontsize=10,
        color=_NEWS,
        ha="center",
    )
    ax.annotate(
        "CLOB liquidation\nexits below equity",
        xy=(0.82, clob_exit_price),
        xytext=(0.58, 0.24),
        arrowprops=dict(arrowstyle="->", lw=1.6, color=_CLOB),
        fontsize=10,
        color=_CLOB,
        ha="center",
    )
    ax.annotate(
        "PM-DFBA exit\nwithin buffer",
        xy=(0.92, pm_dfba_exit_price),
        xytext=(0.78, 0.58),
        arrowprops=dict(arrowstyle="->", lw=1.6, color=_PM_DFBA),
        fontsize=10,
        color=_PM_DFBA,
        ha="center",
    )
    ax.text(
        0.06,
        0.31,
        "bad debt region:\ncollateral exhausted",
        color=_ZERO_EQUITY,
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff1f0", edgecolor="#fecdca"),
    )

    ax.set_title("Simplified Leveraged YES Episode", fontsize=15, fontweight="bold", color=_INK)
    ax.set_xlabel("episode time", fontsize=11)
    ax.set_ylabel("YES probability / liquidation price", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.25, 0.78)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    caption = (
        "p0 = 0.60, leverage = 3x, maintenance buffer = 0.05. "
        "Zero-equity price is where trader collateral is exhausted.\n"
        "Liquidation barrier is where the risk engine tries to exit; "
        "exit price is the actual executable liquidation price."
    )
    fig.text(0.5, 0.025, caption, ha="center", fontsize=8.8, color=_MUTED)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pm_dfba_state_machine(path: str | Path) -> None:
    fig = plt.figure(figsize=(13.2, 7.2))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 1.05))
    ax = fig.add_subplot(grid[0, 0])
    table_ax = fig.add_subplot(grid[0, 1])
    ax.set_title("PM-DFBA State-Contingent Operation", fontsize=15, fontweight="bold", color=_INK)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    nodes = {
        "Normal": (1.2, 4.1),
        "Scheduled\nCatalyst Call": (3.9, 6.25),
        "Volatility\nCall": (3.9, 4.1),
        "Post-Jump\nStabilization": (6.9, 4.1),
        "Resolution /\nReduce-Only": (6.9, 6.25),
        "Thin Market /\nNo-New-Leverage": (6.9, 1.75),
    }
    colors = {
        "Normal": "#eff8ff",
        "Scheduled\nCatalyst Call": "#fff7ed",
        "Volatility\nCall": "#fef3f2",
        "Post-Jump\nStabilization": "#ecfdf3",
        "Resolution /\nReduce-Only": "#f4f3ff",
        "Thin Market /\nNo-New-Leverage": "#f8fafc",
    }
    for label, (x, y) in nodes.items():
        _draw_state_node(ax, x, y, label, colors[label])

    _state_arrow(ax, nodes["Normal"], nodes["Scheduled\nCatalyst Call"], "known event time")
    _state_arrow(
        ax,
        nodes["Normal"],
        nodes["Volatility\nCall"],
        "indicative move\nexceeds threshold",
        label_offset=0.55,
    )
    _state_arrow(
        ax,
        nodes["Volatility\nCall"],
        nodes["Post-Jump\nStabilization"],
        "auction clears /\nimbalance stabilizes",
        label_offset=0.65,
    )
    _state_arrow(
        ax,
        nodes["Post-Jump\nStabilization"],
        nodes["Normal"],
        "depth and spread recover",
        bend=0.18,
        label_offset=-0.55,
    )
    _state_arrow(ax, nodes["Scheduled\nCatalyst Call"], nodes["Resolution /\nReduce-Only"], "close to settlement", dashed=True)
    _state_arrow(ax, nodes["Volatility\nCall"], nodes["Resolution /\nReduce-Only"], "any state:\nclose to settlement", dashed=True)
    _state_arrow(ax, nodes["Post-Jump\nStabilization"], nodes["Thin Market /\nNo-New-Leverage"], "any state:\ndepth/backstop low", dashed=True)
    _state_arrow(ax, nodes["Normal"], nodes["Thin Market /\nNo-New-Leverage"], "depth/backstop\nbelow threshold", dashed=True)

    ax.text(
        4.8,
        0.35,
        "Concept figure only: state transitions are mechanism-design logic, not empirical estimates.",
        ha="center",
        fontsize=9,
        color=_MUTED,
    )

    table_ax.axis("off")
    table_ax.set_title("State Policy Sketch", fontsize=13, fontweight="bold", color=_INK, pad=10)
    rows = [
        ["Normal", "short frequent batches", "normal margin"],
        ["Volatility Call", "extended batch", "pause new leverage / collars"],
        ["Stabilization", "shorter batches", "risk-reducing flow"],
        ["Resolution", "call-only or reduce-only", "no new leverage"],
        ["Thin Market", "restricted clearing", "lower/no leverage"],
    ]
    table = table_ax.table(
        cellText=rows,
        colLabels=["State", "Batch behavior", "Margin behavior"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.24, 0.36, 0.40],
        bbox=[0.0, 0.08, 1.0, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5dd")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#344054")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f9fafb")
            cell.get_text().set_color(_INK)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _draw_timeline_axis(ax: Axes, title: str, color: str) -> None:
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=color)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 100, 250, 500])
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#98a2b3")
    ax.grid(axis="x", alpha=0.18)
    ax.hlines(0.50, 0, 500, color="#98a2b3", linewidth=2)


def _draw_event(ax: Axes, x: float, label: str, color: str, y: float) -> None:
    ax.vlines(x, 0.23, 0.77, color=color, linewidth=1.8, alpha=0.95)
    ax.scatter([x], [0.50], s=72, color=color, zorder=4)
    vertical_alignment = "bottom" if y >= 0.5 else "top"
    ax.text(x, y, label, ha="center", va=vertical_alignment, fontsize=9, color=color)


def _draw_arrow(ax: Axes, x0: float, y0: float, x1: float, y1: float, color: str, label: str) -> None:
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.5,
        color=color,
        alpha=0.9,
    )
    ax.add_patch(arrow)
    ax.text((x0 + x1) / 2, y0 + 0.03, label, ha="center", va="bottom", fontsize=8, color=color)


def _draw_state_node(ax: Axes, x: float, y: float, label: str, facecolor: str) -> None:
    node = FancyBboxPatch(
        (x - 1.0, y - 0.45),
        2.0,
        0.9,
        boxstyle="round,pad=0.08,rounding_size=0.13",
        facecolor=facecolor,
        edgecolor="#344054",
        linewidth=1.4,
        zorder=3,
    )
    ax.add_patch(node)
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5, fontweight="bold", color=_INK, zorder=4)


def _state_arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    dashed: bool = False,
    bend: float = 0.0,
    label_offset: float | None = None,
) -> None:
    x0, y0 = start
    x1, y1 = end
    style = "arc3,rad=" + str(bend)
    arrow = FancyArrowPatch(
        (x0 + 0.95 if x1 >= x0 else x0 - 0.95, y0),
        (x1 - 0.95 if x1 >= x0 else x1 + 0.95, y1),
        arrowstyle="->",
        mutation_scale=12,
        linewidth=1.25,
        linestyle="--" if dashed else "-",
        color="#475467",
        connectionstyle=style,
        zorder=2,
    )
    ax.add_patch(arrow)
    if label_offset is None:
        label_offset = 0.32 if y1 >= y0 else -0.32
    ax.text(
        (x0 + x1) / 2,
        (y0 + y1) / 2 + label_offset,
        label,
        ha="center",
        va="center",
        fontsize=7.6,
        color="#475467",
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.86),
    )

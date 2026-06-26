from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from pm_dfba_sim.metrics import expected_shortfall
from pm_dfba_sim.probability import generate_probability_jump
from pm_dfba_sim.simulation import simulate_venue_trial
from pm_dfba_sim.types import MarketConfig, VenueType
from pm_dfba_sim.venues import implied_clob_race_probability


class AblationMode(str, Enum):
    CLOB = "CLOB"
    FBA = "FBA"
    DFBA = "DFBA"
    PM_DFBA_FULL = "PM_DFBA_FULL"
    PM_DFBA_NO_VOL_CALL = "PM_DFBA_NO_VOL_CALL"
    PM_DFBA_NO_BACKSTOP_ONLY = "PM_DFBA_NO_BACKSTOP_ONLY"
    PM_DFBA_NO_MAKER_TAKER_SEGREGATION = "PM_DFBA_NO_MAKER_TAKER_SEGREGATION"
    PM_DFBA_TOXIC_ONLY = "PM_DFBA_TOXIC_ONLY"
    PM_DFBA_TOXIC_WITH_BACKSTOP = "PM_DFBA_TOXIC_WITH_BACKSTOP"
    PM_DFBA_ADVERSE_DEPTH_ONLY = "PM_DFBA_ADVERSE_DEPTH_ONLY"
    PM_DFBA_ADVERSE_STACK = "PM_DFBA_ADVERSE_STACK"
    TERMINAL_JUMP_STRESS = "TERMINAL_JUMP_STRESS"
    PM_DFBA_NO_BACKSTOP = "PM_DFBA_NO_BACKSTOP_ONLY"
    PM_DFBA_TOXIC_FLOW_MISCLASSIFICATION = "PM_DFBA_ADVERSE_STACK"


@dataclass(frozen=True)
class ScenarioSpec:
    mode: AblationMode
    venue: VenueType
    config: MarketConfig
    description: str


BASELINE_MODES = (
    AblationMode.CLOB,
    AblationMode.FBA,
    AblationMode.DFBA,
    AblationMode.PM_DFBA_FULL,
    AblationMode.PM_DFBA_NO_VOL_CALL,
    AblationMode.PM_DFBA_NO_BACKSTOP_ONLY,
    AblationMode.PM_DFBA_NO_MAKER_TAKER_SEGREGATION,
    AblationMode.PM_DFBA_TOXIC_ONLY,
    AblationMode.PM_DFBA_TOXIC_WITH_BACKSTOP,
    AblationMode.PM_DFBA_ADVERSE_DEPTH_ONLY,
    AblationMode.PM_DFBA_ADVERSE_STACK,
    AblationMode.TERMINAL_JUMP_STRESS,
)
PUBLIC_SWEEP_MODES = (
    AblationMode.CLOB,
    AblationMode.FBA,
    AblationMode.DFBA,
    AblationMode.PM_DFBA_FULL,
    AblationMode.PM_DFBA_NO_VOL_CALL,
    AblationMode.PM_DFBA_TOXIC_ONLY,
    AblationMode.PM_DFBA_ADVERSE_STACK,
)
PRIVATE_SWEEP_MODES = (
    AblationMode.CLOB,
    AblationMode.PM_DFBA_FULL,
    AblationMode.PM_DFBA_TOXIC_ONLY,
    AblationMode.PM_DFBA_ADVERSE_STACK,
)
TERMINAL_SWEEP_MODES = (
    AblationMode.CLOB,
    AblationMode.FBA,
    AblationMode.DFBA,
    AblationMode.PM_DFBA_FULL,
)
BATCH_SWEEP_MODES = (
    AblationMode.CLOB,
    AblationMode.FBA,
    AblationMode.DFBA,
    AblationMode.PM_DFBA_FULL,
    AblationMode.PM_DFBA_NO_VOL_CALL,
)
LATENCY_SWEEP_MODES = (
    AblationMode.CLOB,
    AblationMode.PM_DFBA_FULL,
)


def scenario_spec(base_config: MarketConfig, mode: AblationMode) -> ScenarioSpec:
    config = base_config
    venue = VenueType.PM_DFBA
    description = "Baseline PM-DFBA with volatility-call protection and backstop depth."

    if mode == AblationMode.CLOB:
        venue = VenueType.CLOB
        description = "Continuous limit order book baseline."
    elif mode == AblationMode.FBA:
        venue = VenueType.FBA
        description = "Frequent batch auction without maker/taker segregation."
    elif mode == AblationMode.DFBA:
        venue = VenueType.DFBA
        description = "Dual-flow batch auction without prediction-market-specific protections."
    elif mode == AblationMode.PM_DFBA_FULL:
        pass
    elif mode == AblationMode.PM_DFBA_NO_VOL_CALL:
        config = replace(
            config,
            volatility_call_enabled=False,
            pm_dfba_public_stale_loss_multiplier=config.pm_dfba_private_stale_loss_multiplier,
        )
        description = "PM-DFBA with public-jump volatility-call stale-loss benefit and delay cost removed."
    elif mode in {AblationMode.PM_DFBA_NO_BACKSTOP_ONLY, AblationMode.PM_DFBA_NO_BACKSTOP}:
        config = replace(config, backstop_depth_multiplier=0.0)
        description = "PM-DFBA with only committed backstop depth removed."
    elif mode == AblationMode.PM_DFBA_NO_MAKER_TAKER_SEGREGATION:
        config = replace(
            config,
            pm_dfba_depth_multiplier=config.fba_depth_multiplier,
            pm_dfba_public_stale_loss_multiplier=config.fba_stale_loss_multiplier,
            pm_dfba_private_stale_loss_multiplier=config.fba_stale_loss_multiplier,
            pm_dfba_liquidation_slippage_multiplier=config.fba_liquidation_slippage_multiplier,
        )
        description = "PM-DFBA with maker/taker segregation weakened to FBA-like assumptions."
    elif mode == AblationMode.PM_DFBA_TOXIC_ONLY:
        config = replace(
            config,
            pm_dfba_public_stale_loss_multiplier=(
                config.toxic_flow_public_stale_loss_multiplier
            ),
            pm_dfba_private_stale_loss_multiplier=(
                config.toxic_flow_private_stale_loss_multiplier
            ),
        )
        description = "PM-DFBA with only toxic-flow stale-loss classification degraded."
    elif mode == AblationMode.PM_DFBA_TOXIC_WITH_BACKSTOP:
        config = replace(
            config,
            pm_dfba_depth_multiplier=config.clob_depth_multiplier,
            pm_dfba_public_stale_loss_multiplier=(
                config.toxic_flow_public_stale_loss_multiplier
            ),
            pm_dfba_private_stale_loss_multiplier=(
                config.toxic_flow_private_stale_loss_multiplier
            ),
            pm_dfba_liquidation_slippage_multiplier=(
                config.toxic_flow_liquidation_slippage_multiplier
                * config.clob_liquidation_slippage_multiplier
            ),
        )
        description = "PM-DFBA toxic-flow plus adverse depth stress, with backstop retained."
    elif mode == AblationMode.PM_DFBA_ADVERSE_DEPTH_ONLY:
        config = replace(
            config,
            pm_dfba_depth_multiplier=config.clob_depth_multiplier,
            pm_dfba_liquidation_slippage_multiplier=(
                config.toxic_flow_liquidation_slippage_multiplier
                * config.clob_liquidation_slippage_multiplier
            ),
        )
        description = "PM-DFBA with only primary depth/slippage weakened; stale protection and backstop remain."
    elif mode in {
        AblationMode.PM_DFBA_ADVERSE_STACK,
        AblationMode.PM_DFBA_TOXIC_FLOW_MISCLASSIFICATION,
    }:
        config = replace(
            config,
            pm_dfba_depth_multiplier=config.clob_depth_multiplier,
            backstop_depth_multiplier=0.0,
            pm_dfba_public_stale_loss_multiplier=(
                config.toxic_flow_public_stale_loss_multiplier
            ),
            pm_dfba_private_stale_loss_multiplier=(
                config.toxic_flow_private_stale_loss_multiplier
            ),
            pm_dfba_liquidation_slippage_multiplier=(
                config.toxic_flow_liquidation_slippage_multiplier
                * config.clob_liquidation_slippage_multiplier
            ),
        )
        description = "PM-DFBA combined worst case: toxic classification, adverse depth/slippage, and no backstop."
    elif mode == AblationMode.TERMINAL_JUMP_STRESS:
        config = replace(config, terminal_jump_probability=max(config.terminal_jump_probability, 0.25))
        description = "PM-DFBA under elevated terminal instant-resolution jump probability."
    else:
        raise ValueError(f"unsupported ablation mode: {mode}")

    return ScenarioSpec(mode=mode, venue=venue, config=config, description=description)


def simulate_mode(base_config: MarketConfig, mode: AblationMode) -> pd.DataFrame:
    spec = scenario_spec(base_config, mode)
    rng = np.random.default_rng(spec.config.seed)
    rows: list[dict[str, object]] = []

    for trial_id in range(spec.config.n_trials):
        event = generate_probability_jump(spec.config, rng)
        maker_latency_ms = float(rng.exponential(spec.config.maker_latency_mean_ms))
        taker_latency_ms = float(rng.exponential(spec.config.taker_latency_mean_ms))

        for leverage in spec.config.leverage_values:
            result = simulate_venue_trial(
                config=spec.config,
                event=event,
                maker_latency_ms=maker_latency_ms,
                taker_latency_ms=taker_latency_ms,
                trial_id=trial_id,
                leverage=leverage,
                venue=spec.venue,
            )
            row = result.to_row()
            row["mode"] = spec.mode.value
            row["scenario_description"] = spec.description
            row["private_jump"] = (not result.public_jump) and (not result.terminal_jump)
            row["maker_latency_ms"] = maker_latency_ms
            row["taker_latency_ms"] = taker_latency_ms
            row["clob_race"] = taker_latency_ms < maker_latency_ms
            rows.append(row)

    return pd.DataFrame(rows)


def summarize_trial_frame(trials: pd.DataFrame, group_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    grouping = list(group_cols) + ["mode", "venue", "leverage"]
    rows: list[dict[str, object]] = []

    for keys, group in trials.groupby(grouping, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        bad_debt = group["bad_debt"]
        row.update(
            {
                "n_trials": int(group["trial_id"].nunique()),
                "public_jump_rate": float(group["public_jump"].mean()),
                "private_jump_rate": float(group["private_jump"].mean()),
                "terminal_jump_rate": float(group["terminal_jump"].mean()),
                "bad_debt_probability": float((bad_debt > 0).mean()),
                "bad_debt_mean": float(bad_debt.mean()),
                "bad_debt_expected_shortfall_95": expected_shortfall(bad_debt, 0.95),
                "bad_debt_expected_shortfall_99": expected_shortfall(bad_debt, 0.99),
                "liquidation_shortfall_mean": float(group["liquidation_shortfall"].mean()),
                "stale_quote_loss_mean": float(group["stale_quote_loss"].mean()),
                "public_stale_quote_loss_mean": float(group["public_stale_quote_loss"].mean()),
                "maker_loss_placeholder_mean": float(
                    group["maker_loss_placeholder"].mean()
                ),
                "liquidation_trigger_rate": float(group["liquidation_triggered"].mean()),
                "effective_liquidation_depth": float(group["effective_liquidation_depth"].mean()),
                "liquidation_unfilled_quantity_mean": float(
                    group["liquidation_unfilled_quantity"].mean()
                ),
                "liquidation_collar_breach_rate": float(
                    group["liquidation_collar_breached"].mean()
                ),
                "liquidation_used_backstop_depth_mean": float(
                    group["liquidation_used_backstop_depth"].mean()
                ),
                "taker_delay_cost": float(group["taker_delay_cost"].mean()),
                "stale_plus_delay_cost_mean": float(
                    group["stale_quote_loss"].mean() + group["taker_delay_cost"].mean()
                ),
                "clob_race_probability": float(group["clob_race"].mean()),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(grouping).reset_index(drop=True)


def safe_leverage_by_group(
    summary: pd.DataFrame,
    tolerance: float,
    group_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    grouping = list(group_cols) + ["mode", "venue"]
    rows: list[dict[str, object]] = []

    for keys, group in summary.groupby(grouping, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(grouping, keys))
        safe = group[group["bad_debt_probability"] <= tolerance]
        row["bad_debt_tolerance"] = tolerance
        row["safe_leverage_at_bad_debt_tolerance"] = (
            None if safe.empty else float(safe["leverage"].max())
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values(grouping).reset_index(drop=True)


def run_baseline_ablation(config: MarketConfig) -> pd.DataFrame:
    trials = pd.concat([simulate_mode(config, mode) for mode in BASELINE_MODES], ignore_index=True)
    return summarize_trial_frame(trials)


def run_public_jump_share_sweep(config: MarketConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for share in config.public_jump_share_sweep:
        sweep_config = replace(
            config,
            terminal_jump_probability=0.0,
            public_jump_probability=share,
        )
        for mode in PUBLIC_SWEEP_MODES:
            frame = simulate_mode(sweep_config, mode)
            frame["public_jump_share"] = share
            frame["private_jump_share"] = 1.0 - share
            frames.append(frame)

    summary = summarize_trial_frame(pd.concat(frames, ignore_index=True), ("public_jump_share",))
    safe = safe_leverage_by_group(summary, config.bad_debt_tolerance, ("public_jump_share",))
    return summary, safe


def run_private_jump_share_sweep(config: MarketConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for share in config.private_jump_share_sweep:
        sweep_config = replace(
            config,
            terminal_jump_probability=0.0,
            public_jump_probability=1.0 - share,
        )
        for mode in PRIVATE_SWEEP_MODES:
            frame = simulate_mode(sweep_config, mode)
            frame["private_jump_share"] = share
            frames.append(frame)

    return summarize_trial_frame(pd.concat(frames, ignore_index=True), ("private_jump_share",))


def run_backstop_depth_sweep(config: MarketConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for share in config.backstop_depth_share_sweep:
        multiplier = share * config.quantity / max(config.base_liquidation_depth, 1e-9)
        sweep_config = replace(config, backstop_depth_multiplier=multiplier)
        frame = simulate_mode(sweep_config, AblationMode.PM_DFBA_FULL)
        frame["backstop_depth_share"] = share
        frames.append(frame)

    summary = summarize_trial_frame(pd.concat(frames, ignore_index=True), ("backstop_depth_share",))
    return summary


def run_batch_interval_sweep(config: MarketConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for interval_ms in config.batch_interval_ms_sweep:
        sweep_config = replace(config, terminal_jump_probability=0.0, batch_interval_ms=interval_ms)
        for mode in BATCH_SWEEP_MODES:
            frame = simulate_mode(sweep_config, mode)
            frame["batch_interval_ms"] = interval_ms
            frames.append(frame)

    return summarize_trial_frame(pd.concat(frames, ignore_index=True), ("batch_interval_ms",))


def run_terminal_jump_stress(config: MarketConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for terminal_probability in config.terminal_jump_probability_sweep:
        sweep_config = replace(config, terminal_jump_probability=terminal_probability)
        for mode in TERMINAL_SWEEP_MODES:
            frame = simulate_mode(sweep_config, mode)
            frame["terminal_jump_probability"] = terminal_probability
            frames.append(frame)

    return summarize_trial_frame(
        pd.concat(frames, ignore_index=True),
        ("terminal_jump_probability",),
    )


def run_latency_sweep(config: MarketConfig) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for maker_latency_mean_ms in config.maker_latency_mean_ms_sweep:
        for taker_latency_mean_ms in config.taker_latency_mean_ms_sweep:
            sweep_config = replace(
                config,
                terminal_jump_probability=0.0,
                maker_latency_mean_ms=maker_latency_mean_ms,
                taker_latency_mean_ms=taker_latency_mean_ms,
            )
            race_probability = implied_clob_race_probability(
                maker_latency_mean_ms,
                taker_latency_mean_ms,
            )
            for mode in LATENCY_SWEEP_MODES:
                frame = simulate_mode(sweep_config, mode)
                frame["maker_latency_mean_ms"] = maker_latency_mean_ms
                frame["taker_latency_mean_ms"] = taker_latency_mean_ms
                frame["implied_clob_race_probability"] = race_probability
                frames.append(frame)

    group_cols = (
        "maker_latency_mean_ms",
        "taker_latency_mean_ms",
        "implied_clob_race_probability",
    )
    summary = summarize_trial_frame(pd.concat(frames, ignore_index=True), group_cols)
    return _add_stale_loss_advantage_vs_clob(summary, group_cols)


def _add_stale_loss_advantage_vs_clob(
    summary: pd.DataFrame,
    group_cols: tuple[str, ...],
) -> pd.DataFrame:
    key_cols = list(group_cols) + ["leverage"]
    clob_stale = summary[summary["mode"] == AblationMode.CLOB.value][
        key_cols + ["stale_quote_loss_mean"]
    ].rename(columns={"stale_quote_loss_mean": "clob_stale_quote_loss_mean"})
    merged = summary.merge(clob_stale, on=key_cols, how="left")
    merged["stale_loss_advantage_vs_clob"] = (
        merged["clob_stale_quote_loss_mean"] - merged["stale_quote_loss_mean"]
    )
    return merged


def run_ablation_suite(config: MarketConfig, out_dir: str | Path) -> dict[str, pd.DataFrame]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ablation_summary = run_baseline_ablation(config)
    public_summary, safe_public = run_public_jump_share_sweep(config)
    backstop_summary = run_backstop_depth_sweep(config)
    batch_summary = run_batch_interval_sweep(config)
    private_summary = run_private_jump_share_sweep(config)
    terminal_summary = run_terminal_jump_stress(config)
    latency_summary = run_latency_sweep(config)

    ablation_summary.to_csv(out_path / "ablation_summary.csv", index=False)
    safe_public.to_csv(out_path / "safe_leverage_by_public_jump_share.csv", index=False)
    backstop_summary.to_csv(out_path / "bad_debt_by_backstop_depth.csv", index=False)
    batch_summary.to_csv(out_path / "stale_loss_by_batch_interval.csv", index=False)
    terminal_summary.to_csv(out_path / "terminal_jump_stress.csv", index=False)
    latency_summary.to_csv(out_path / "latency_sweep.csv", index=False)

    return {
        "ablation_summary": ablation_summary,
        "public_jump_summary": public_summary,
        "safe_leverage_by_public_jump_share": safe_public,
        "bad_debt_by_backstop_depth": backstop_summary,
        "stale_loss_by_batch_interval": batch_summary,
        "private_information_stress": private_summary,
        "terminal_jump_stress": terminal_summary,
        "latency_sweep": latency_summary,
    }

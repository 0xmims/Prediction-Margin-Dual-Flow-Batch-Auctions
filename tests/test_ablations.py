from dataclasses import replace

import pytest

from pm_dfba_sim.ablations import (
    AblationMode,
    run_ablation_suite,
    run_backstop_depth_sweep,
    run_private_jump_share_sweep,
    run_public_jump_share_sweep,
    simulate_mode,
    summarize_trial_frame,
)
from pm_dfba_sim.run_ablations import write_ablation_figures
from pm_dfba_sim.simulation import load_config


def _small_config(**overrides):
    values = {
        "n_trials": 120,
        "leverage_values": (1.25, 1.5, 2.0, 3.0, 5.0),
        **overrides,
    }
    return replace(
        load_config("configs/baseline.json"),
        **values,
    )


def _summary_for(config, mode):
    return summarize_trial_frame(simulate_mode(config, mode))


def _metric(summary, mode, leverage, metric):
    row = summary[(summary["mode"] == mode.value) & (summary["leverage"] == leverage)]
    assert len(row) == 1
    return float(row.iloc[0][metric])


def test_terminal_jumps_can_create_bad_debt_under_pm_dfba():
    config = _small_config(
        terminal_jump_probability=1.0,
        adverse_jump_probability=1.0,
        public_jump_probability=0.0,
    )
    summary = _summary_for(config, AblationMode.PM_DFBA_FULL)

    assert _metric(summary, AblationMode.PM_DFBA_FULL, 3.0, "bad_debt_probability") == 1.0


def test_no_backstop_has_worse_or_equal_liquidation_performance_than_full_pm_dfba():
    config = _small_config(n_trials=160)
    full = _summary_for(config, AblationMode.PM_DFBA_FULL)
    no_backstop = _summary_for(config, AblationMode.PM_DFBA_NO_BACKSTOP)

    assert _metric(
        no_backstop,
        AblationMode.PM_DFBA_NO_BACKSTOP,
        3.0,
        "liquidation_shortfall_mean",
    ) >= _metric(full, AblationMode.PM_DFBA_FULL, 3.0, "liquidation_shortfall_mean")


def test_no_vol_call_increases_public_stale_loss():
    config = _small_config(
        terminal_jump_probability=0.0,
        public_jump_probability=1.0,
    )
    full = _summary_for(config, AblationMode.PM_DFBA_FULL)
    no_vol = _summary_for(config, AblationMode.PM_DFBA_NO_VOL_CALL)

    assert _metric(
        no_vol,
        AblationMode.PM_DFBA_NO_VOL_CALL,
        3.0,
        "public_stale_quote_loss_mean",
    ) > _metric(full, AblationMode.PM_DFBA_FULL, 3.0, "public_stale_quote_loss_mean")


def test_increasing_private_jump_share_reduces_pm_dfba_stale_loss_advantage():
    config = _small_config(
        n_trials=160,
        private_jump_share_sweep=(0.0, 1.0),
    )
    summary = run_private_jump_share_sweep(config)
    leverage = 3.0

    def advantage(private_share):
        rows = summary[
            (summary["private_jump_share"] == private_share)
            & (summary["leverage"] == leverage)
        ]
        clob = rows[rows["mode"] == AblationMode.CLOB.value].iloc[0]
        pm = rows[rows["mode"] == AblationMode.PM_DFBA_FULL.value].iloc[0]
        return clob["stale_quote_loss_mean"] - pm["stale_quote_loss_mean"]

    assert advantage(1.0) < advantage(0.0)


def test_toxic_flow_misclassification_reduces_pm_dfba_advantage():
    config = _small_config(
        n_trials=160,
        terminal_jump_probability=0.0,
        public_jump_probability=0.75,
    )
    clob = _summary_for(config, AblationMode.CLOB)
    full = _summary_for(config, AblationMode.PM_DFBA_FULL)
    toxic = _summary_for(config, AblationMode.PM_DFBA_TOXIC_FLOW_MISCLASSIFICATION)
    leverage = 3.0

    full_advantage = _metric(clob, AblationMode.CLOB, leverage, "stale_quote_loss_mean") - _metric(
        full,
        AblationMode.PM_DFBA_FULL,
        leverage,
        "stale_quote_loss_mean",
    )
    toxic_advantage = _metric(clob, AblationMode.CLOB, leverage, "stale_quote_loss_mean") - _metric(
        toxic,
        AblationMode.PM_DFBA_TOXIC_FLOW_MISCLASSIFICATION,
        leverage,
        "stale_quote_loss_mean",
    )

    assert toxic_advantage < full_advantage


def test_higher_leverage_increases_bad_debt_probability_for_core_venues():
    config = _small_config(n_trials=160)
    for mode in [
        AblationMode.CLOB,
        AblationMode.FBA,
        AblationMode.DFBA,
        AblationMode.PM_DFBA_FULL,
    ]:
        summary = _summary_for(config, mode).sort_values("leverage")
        bad_debt_probabilities = summary["bad_debt_probability"].to_list()
        assert bad_debt_probabilities == sorted(bad_debt_probabilities)


def test_ablation_outputs_are_generated_and_non_empty(tmp_path):
    config = _small_config(
        n_trials=8,
        leverage_values=(1.25, 3.0),
        public_jump_share_sweep=(0.0, 1.0),
        private_jump_share_sweep=(0.0, 1.0),
        terminal_jump_probability_sweep=(0.0, 0.25),
        batch_interval_ms_sweep=(50.0, 250.0),
        backstop_depth_share_sweep=(0.0, 0.5),
    )

    outputs = run_ablation_suite(config, tmp_path)
    write_ablation_figures(outputs, tmp_path, plot_leverage=3.0)

    expected_files = [
        "ablation_summary.csv",
        "safe_leverage_by_public_jump_share.csv",
        "bad_debt_by_backstop_depth.csv",
        "stale_loss_by_batch_interval.csv",
        "terminal_jump_stress.csv",
        "safe_leverage_vs_public_jump_share.png",
        "bad_debt_by_backstop_depth.png",
        "stale_loss_by_batch_interval.png",
        "private_information_stress.png",
        "terminal_jump_failure.png",
    ]
    for name in expected_files:
        path = tmp_path / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_public_and_backstop_sweeps_are_non_empty():
    config = _small_config(
        n_trials=8,
        leverage_values=(1.25, 3.0),
        public_jump_share_sweep=(0.0, 1.0),
        backstop_depth_share_sweep=(0.0, 0.5),
    )

    _, safe_public = run_public_jump_share_sweep(config)
    backstop = run_backstop_depth_sweep(config)

    assert not safe_public.empty
    assert not backstop.empty

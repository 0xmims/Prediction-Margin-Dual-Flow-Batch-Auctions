from dataclasses import replace

import pandas as pd
import pytest

from pm_dfba_sim.ablations import (
    AblationMode,
    run_ablation_suite,
    run_backstop_depth_sweep,
    run_latency_sweep,
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


def test_ablation_mode_reproducibility_with_same_seed():
    config = _small_config(n_trials=40, leverage_values=(1.25, 3.0))

    first = simulate_mode(config, AblationMode.PM_DFBA_FULL)
    second = simulate_mode(config, AblationMode.PM_DFBA_FULL)

    pd.testing.assert_frame_equal(first, second)


def test_zero_jump_scenario_produces_zero_stale_loss_and_bad_debt():
    config = _small_config(
        jump_probability=0.0,
        terminal_jump_probability=0.0,
    )
    frames = [
        simulate_mode(config, AblationMode.CLOB),
        simulate_mode(config, AblationMode.PM_DFBA_FULL),
    ]
    summary = summarize_trial_frame(pd.concat(frames, ignore_index=True))

    assert summary["stale_quote_loss_mean"].max() == 0.0
    assert summary["bad_debt_probability"].max() == 0.0
    assert summary["bad_debt_mean"].max() == 0.0


def test_no_backstop_has_worse_or_equal_liquidation_performance_than_full_pm_dfba():
    config = _small_config(n_trials=160)
    full = _summary_for(config, AblationMode.PM_DFBA_FULL)
    no_backstop = _summary_for(config, AblationMode.PM_DFBA_NO_BACKSTOP_ONLY)

    assert _metric(
        no_backstop,
        AblationMode.PM_DFBA_NO_BACKSTOP_ONLY,
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


def test_no_vol_call_removes_public_jump_taker_delay_cost():
    config = _small_config(
        terminal_jump_probability=0.0,
        public_jump_probability=1.0,
        jump_size_min=0.20,
        jump_size_max=0.20,
    )
    full = _summary_for(config, AblationMode.PM_DFBA_FULL)
    no_vol = _summary_for(config, AblationMode.PM_DFBA_NO_VOL_CALL)

    assert _metric(
        no_vol,
        AblationMode.PM_DFBA_NO_VOL_CALL,
        3.0,
        "taker_delay_cost",
    ) <= _metric(full, AblationMode.PM_DFBA_FULL, 3.0, "taker_delay_cost")


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
    toxic = _summary_for(config, AblationMode.PM_DFBA_TOXIC_ONLY)
    leverage = 3.0

    full_advantage = _metric(clob, AblationMode.CLOB, leverage, "stale_quote_loss_mean") - _metric(
        full,
        AblationMode.PM_DFBA_FULL,
        leverage,
        "stale_quote_loss_mean",
    )
    toxic_advantage = _metric(clob, AblationMode.CLOB, leverage, "stale_quote_loss_mean") - _metric(
        toxic,
        AblationMode.PM_DFBA_TOXIC_ONLY,
        leverage,
        "stale_quote_loss_mean",
    )

    assert toxic_advantage < full_advantage


def test_latency_sweep_reduces_pm_dfba_advantage_when_clob_race_probability_falls():
    config = _small_config(
        n_trials=240,
        leverage_values=(3.0,),
        terminal_jump_probability=0.0,
        public_jump_probability=1.0,
        jump_size_min=0.20,
        jump_size_max=0.20,
        maker_latency_mean_ms_sweep=(20.0, 200.0),
        taker_latency_mean_ms_sweep=(20.0, 200.0),
    )

    summary = run_latency_sweep(config)
    pm = summary[
        (summary["mode"] == AblationMode.PM_DFBA_FULL.value)
        & (summary["leverage"] == 3.0)
    ]
    low_race = pm.loc[pm["implied_clob_race_probability"].idxmin()]
    high_race = pm.loc[pm["implied_clob_race_probability"].idxmax()]

    assert high_race["stale_loss_advantage_vs_clob"] > low_race["stale_loss_advantage_vs_clob"]


def test_toxic_variants_isolate_loss_drivers():
    config = _small_config(
        n_trials=180,
        terminal_jump_probability=0.0,
        public_jump_probability=0.75,
    )
    leverage = 3.0
    full = _summary_for(config, AblationMode.PM_DFBA_FULL)
    toxic_only = _summary_for(config, AblationMode.PM_DFBA_TOXIC_ONLY)
    toxic_with_backstop = _summary_for(config, AblationMode.PM_DFBA_TOXIC_WITH_BACKSTOP)
    no_backstop = _summary_for(config, AblationMode.PM_DFBA_NO_BACKSTOP_ONLY)
    adverse_depth = _summary_for(config, AblationMode.PM_DFBA_ADVERSE_DEPTH_ONLY)
    adverse_stack = _summary_for(config, AblationMode.PM_DFBA_ADVERSE_STACK)

    full_stale = _metric(full, AblationMode.PM_DFBA_FULL, leverage, "stale_quote_loss_mean")
    full_shortfall = _metric(
        full,
        AblationMode.PM_DFBA_FULL,
        leverage,
        "liquidation_shortfall_mean",
    )

    assert _metric(
        toxic_only,
        AblationMode.PM_DFBA_TOXIC_ONLY,
        leverage,
        "stale_quote_loss_mean",
    ) > full_stale
    assert _metric(
        toxic_only,
        AblationMode.PM_DFBA_TOXIC_ONLY,
        leverage,
        "liquidation_shortfall_mean",
    ) == pytest.approx(full_shortfall)
    assert _metric(
        no_backstop,
        AblationMode.PM_DFBA_NO_BACKSTOP_ONLY,
        leverage,
        "stale_quote_loss_mean",
    ) == pytest.approx(full_stale)
    assert _metric(
        no_backstop,
        AblationMode.PM_DFBA_NO_BACKSTOP_ONLY,
        leverage,
        "liquidation_shortfall_mean",
    ) >= full_shortfall
    assert _metric(
        adverse_depth,
        AblationMode.PM_DFBA_ADVERSE_DEPTH_ONLY,
        leverage,
        "stale_quote_loss_mean",
    ) == pytest.approx(full_stale)
    assert _metric(
        adverse_depth,
        AblationMode.PM_DFBA_ADVERSE_DEPTH_ONLY,
        leverage,
        "liquidation_shortfall_mean",
    ) >= full_shortfall
    assert _metric(
        adverse_stack,
        AblationMode.PM_DFBA_ADVERSE_STACK,
        leverage,
        "liquidation_shortfall_mean",
    ) >= _metric(
        toxic_with_backstop,
        AblationMode.PM_DFBA_TOXIC_WITH_BACKSTOP,
        leverage,
        "liquidation_shortfall_mean",
    )


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
        maker_latency_mean_ms_sweep=(20.0, 200.0),
        taker_latency_mean_ms_sweep=(20.0, 200.0),
    )

    outputs = run_ablation_suite(config, tmp_path)
    write_ablation_figures(outputs, tmp_path, plot_leverage=3.0, plot_all_leverages=True)

    expected_files = [
        "ablation_summary.csv",
        "safe_leverage_by_public_jump_share.csv",
        "bad_debt_by_backstop_depth.csv",
        "stale_loss_by_batch_interval.csv",
        "terminal_jump_stress.csv",
        "latency_sweep.csv",
        "safe_leverage_vs_public_jump_share.png",
        "bad_debt_by_backstop_depth.png",
        "stale_loss_by_batch_interval.png",
        "private_information_stress.png",
        "terminal_jump_failure.png",
        "latency_race_probability.png",
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

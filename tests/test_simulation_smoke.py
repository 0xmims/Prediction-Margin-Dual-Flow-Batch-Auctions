from dataclasses import replace

from pm_dfba_sim.simulation import load_config, simulate_experiment
from pm_dfba_sim.types import VenueType


def test_baseline_simulation_produces_all_venues_and_leverages():
    config = replace(load_config("configs/baseline.json"), n_trials=5)

    results = simulate_experiment(config)

    assert results
    assert {result.venue for result in results} == set(VenueType)
    assert {result.leverage for result in results} == set(config.leverage_values)

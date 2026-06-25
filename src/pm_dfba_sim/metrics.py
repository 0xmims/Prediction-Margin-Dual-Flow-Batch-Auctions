from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pm_dfba_sim.types import MarketConfig, TrialResult, VenueType


def expected_shortfall(losses: list[float] | pd.Series, alpha: float) -> float:
    values = np.asarray(losses, dtype=float)
    if values.size == 0:
        return 0.0
    ordered = np.sort(values)
    threshold_index = max(0, math.ceil(alpha * len(ordered)) - 1)
    threshold = ordered[threshold_index]
    tail = ordered[ordered >= threshold]
    if tail.size == 0:
        return float(ordered[-1])
    return float(tail.mean())


def summarize_trials(results: list[TrialResult]) -> pd.DataFrame:
    rows = [result.to_row() for result in results]
    trials = pd.DataFrame(rows)

    summary_rows: list[dict[str, float | str]] = []
    for (venue, leverage), group in trials.groupby(["venue", "leverage"], sort=True):
        bad_debt = group["bad_debt"]
        summary_rows.append(
            {
                "venue": venue,
                "leverage": float(leverage),
                "bad_debt_probability": float((bad_debt > 0).mean()),
                "bad_debt_mean": float(bad_debt.mean()),
                "bad_debt_expected_shortfall_95": expected_shortfall(bad_debt, 0.95),
                "bad_debt_expected_shortfall_99": expected_shortfall(bad_debt, 0.99),
                "liquidation_shortfall_mean": float(group["liquidation_shortfall"].mean()),
                "stale_quote_loss_mean": float(group["stale_quote_loss"].mean()),
                "public_stale_quote_loss_mean": float(group["public_stale_quote_loss"].mean()),
                "maker_loss_mean": float(group["maker_loss"].mean()),
                "liquidation_trigger_rate": float(group["liquidation_triggered"].mean()),
                "effective_liquidation_depth": float(group["effective_liquidation_depth"].mean()),
                "taker_delay_cost": float(group["taker_delay_cost"].mean()),
            }
        )

    return pd.DataFrame(summary_rows).sort_values(["venue", "leverage"]).reset_index(drop=True)


def safe_leverage(summary: pd.DataFrame, config: MarketConfig) -> pd.DataFrame:
    rows: list[dict[str, float | str | None]] = []
    for venue in VenueType:
        venue_summary = summary[summary["venue"] == venue.value].sort_values("leverage")
        safe = venue_summary[
            venue_summary["bad_debt_probability"] <= config.bad_debt_tolerance
        ]
        rows.append(
            {
                "venue": venue.value,
                "bad_debt_tolerance": config.bad_debt_tolerance,
                "safe_leverage_at_bad_debt_tolerance": (
                    None if safe.empty else float(safe["leverage"].max())
                ),
            }
        )
    return pd.DataFrame(rows)

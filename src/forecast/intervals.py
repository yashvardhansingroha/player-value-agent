"""Prediction intervals from EARLIER out-of-sample errors only.

Interval = point forecast x [q_lo, q_hi], where q are empirical quantiles of actual / forecast taken from earlier issued
forecasts (same channel, same kind) whose D30 outcome was already complete and reported at the calibration date.
If fewer than the configured minimum such errors exist, the interval is reported as unavailable with the reason; it is
never filled in with a made-up width.
"""
import numpy as np
import pandas as pd


def interval_from_ratios(ratios, point: float, alpha: float, min_calibration: int) -> dict:
    ratios = np.asarray([r for r in ratios if r is not None and np.isfinite(r)], dtype=float)
    if len(ratios) < min_calibration:
        return {"status": "unavailable", "lo": None, "hi": None, "calib_n": int(len(ratios)),
                "reason": f"insufficient_calibration_history: {len(ratios)} matured earlier forecasts, need {min_calibration}"}
    lo_q, hi_q = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return {"status": "available", "lo": float(point * lo_q), "hi": float(point * hi_q), "calib_n": int(len(ratios)), "reason": None}


def apply_floor(lo, hi, floor: float):
    """Cumulative D30 net revenue cannot be below the D0-D3 net revenue already observed (revenue is non-negative).
    Raise lo to the floor, and hi to at least lo, so the bounds stay ordered. This is a structural consistency correction:
    it never widens an interval (lo only goes up, hi only goes up when it was below lo) and says nothing about coverage.
    Returns (lo, hi, adjusted). Unavailable intervals (None) pass through unchanged."""
    if lo is None or hi is None:
        return lo, hi, False
    new_lo = max(lo, floor)
    new_hi = max(hi, new_lo)
    return new_lo, new_hi, bool(new_lo != lo or new_hi != hi)


def matured_ratios(earlier: pd.DataFrame, outcomes: pd.DataFrame, outcome_complete_at: pd.DataFrame, window: int):
    """earlier: previously issued forecasts with columns channel, install_date, point (>0).
    outcomes: channel, install_date, y30 (ONLY outcomes complete at the calibration date).
    outcome_complete_at: channel, install_date, complete30 (declared completeness date of the D30 milestone).
    Returns (ratios list for the most recent `window` matured forecasts, latest complete30 used)."""
    m = earlier.merge(outcomes, on=["channel", "install_date"], how="inner").merge(
        outcome_complete_at, on=["channel", "install_date"], how="inner")
    m = m[m.point > 0].sort_values("install_date").tail(window)
    if m.empty:
        return [], None
    return (m.y30 / m.point).tolist(), str(m.complete30.max())

"""Baseline D30 forecast: multiplier = sum(D30 net revenue) / sum(D3 net revenue) over a channel's recent, eligible,
D30-complete cohorts ("ratio of sums"). Forecast for a new cohort = multiplier x its D0-D3 net revenue.

Pooling and fallback rules (all explicit, none silent):
  1. channel multiplier  : channel has >= min_train_cohorts eligible training cohorts
  2. pooled multiplier   : otherwise, union of all paid channels' training cohorts if >= min_pooled_cohorts
  3. no forecast         : otherwise ("insufficient_training_history")
  zero D3 revenue        : multiplier is undefined (0 x m = 0 would be misleading), so the forecast is the training
                           average D30 value per install x installs, method = zero_d3_fallback.
Also returns a reference forecast (training average value per install x installs, ignoring D3 revenue) so the value of the
multiplier can be judged.
"""
import numpy as np
import pandas as pd


def select_training(pool: pd.DataFrame, window: int) -> pd.DataFrame:
    """pool: eligible D30-complete cohorts of ONE channel with columns install_date, y3, y30, installs. Most recent `window`."""
    return pool.sort_values("install_date").tail(window)


def fit_multiplier(train: pd.DataFrame) -> dict:
    """Ratio of sums over the training cohorts. Returns None when sum(y3) <= 0."""
    s3, s30, n_inst = float(train.y3.sum()), float(train.y30.sum()), float(train.installs.sum())
    if len(train) == 0 or s3 <= 0:
        return None
    return {
        "n_cohorts": int(len(train)),
        "first_install": str(train.install_date.min()),
        "last_install": str(train.install_date.max()),
        "sum_y3": s3,
        "sum_y30": s30,
        "sum_installs": n_inst,
        "multiplier": s30 / s3,
        "value_per_install": s30 / n_inst if n_inst > 0 else None,
        # latest date at which any training cohort's D30 outcome became complete (must be <= the issuance date)
        "max_outcome_complete_at": str(train.complete30.max()) if "complete30" in train.columns else None,
    }


def choose_training(channel: str, channel_pools: dict, cfg: dict) -> tuple:
    """Return (level, training_info_or_None, detail). channel_pools maps channel -> eligible D30-complete cohorts DataFrame."""
    f = cfg["forecast"]
    own = select_training(channel_pools.get(channel, pd.DataFrame(columns=["install_date", "y3", "y30", "installs"])), f["train_window_cohorts"])
    if len(own) >= f["min_train_cohorts"]:
        info = fit_multiplier(own)
        if info is not None:
            return "channel", info, {}
    pooled = pd.concat([select_training(p, f["train_window_cohorts"]) for p in channel_pools.values() if len(p)] or [own], ignore_index=True)
    if len(pooled) >= f["min_pooled_cohorts"]:
        info = fit_multiplier(pooled)
        if info is not None:
            return "pooled", info, {}
    return None, None, {"channel_training_cohorts": int(len(own)), "pooled_training_cohorts": int(len(pooled)),
                        "needed_channel": f["min_train_cohorts"], "needed_pooled": f["min_pooled_cohorts"]}


def forecast_cohort(y3: float, installs: int, level: str, info: dict) -> dict:
    """Point forecast and reference for one cohort given the training info. Handles zero D3 revenue explicitly."""
    reference = info["value_per_install"] * installs if info["value_per_install"] is not None else None
    if y3 is None or np.isnan(y3) or y3 <= 0:
        method = "zero_d3_fallback"
        point = reference
    else:
        method = "channel_multiplier" if level == "channel" else "pooled_multiplier"
        point = info["multiplier"] * y3
    return {"method": method, "point": point, "reference": reference}

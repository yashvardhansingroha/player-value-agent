"""Economics for the budget review. Pure functions plus small decision-date data loaders. No hidden truth, no outcomes
from after the decision date (every loader goes through the availability layer at `as_of`).

DEFINITIONS (also in the README):
  average D30 ROAS (forecast-based)  = forecast D30 net revenue of the INCLUDED cohorts / spend of those same cohorts
  target D30 ROAS                    = payback_multiple / (sum D180 net / sum D30 net) over mature, reported, eligible cohorts
  model-estimated finite-step marginal D30 ROAS
        = (extra D30 net revenue of the proposed step) / (extra spend of the step), where
          extra installs = f(s1) - f(s0), f(s) = I_ref * (s / s_ref)^b  (log-log response fitted on history),
          extra revenue  = extra installs x forecast value per install x incremental_quality_haircut.
  derivative marginal ROAS (reference only) = value x haircut x b x f(s0) / s0, the slope AT the current budget, not the step.

ASSUMPTIONS the marginal estimate rests on (it is NOT causal incrementality evidence):
  1. the historical spend -> installs association is read as a response curve (no unobserved confounding of spend changes);
  2. the curve holds for a small step INSIDE the observed spend range (steps outside it are refused);
  3. incremental players are worth `incremental_quality_haircut` x the included cohorts' average forecast value per install;
  4. no spillover to organic or other channels.
"""
import numpy as np
import pandas as pd

from src.data.metrics import cohort_outcomes


# ------------------------------------------------------------------ average ROAS


def average_roas(forecast_revenue: float, spend: float):
    """Forecast D30 net revenue / spend over the SAME cohorts. None if spend is not positive."""
    return None if spend is None or spend <= 0 else float(forecast_revenue) / float(spend)


# ------------------------------------------------------------------ payback target


def payback_target(mature: pd.DataFrame, pcfg: dict, channel: str) -> dict:
    """mature: eligible cohorts whose D180 milestone is complete at the decision date, columns
    channel, install_date, installs, y30, y180 (all from availability-filtered data). Returns a dict with the target or the
    reason there is none. Per-channel if the channel has enough mature history, else pooled across paid channels, else none."""
    def fit(df, level):
        df = df.sort_values("install_date").tail(pcfg["trailing_cohorts"])
        n, inst = len(df), int(df.installs.sum())
        if n < pcfg["min_mature_cohorts"] or inst < pcfg["min_mature_installs"]:
            return None, {"n_cohorts": n, "installs": inst}
        s30, s180 = float(df.y30.sum()), float(df.y180.sum())
        if s30 <= 0 or s180 <= 0:
            return None, {"n_cohorts": n, "installs": inst, "note": "non-positive revenue sums"}
        ratio = s180 / s30
        return {"level": level, "n_cohorts": n, "installs": inst, "first_install": str(df.install_date.min()), "last_install": str(df.install_date.max()),
                "sum_d30": s30, "sum_d180": s180, "d180_over_d30": ratio, "target_d30_roas": pcfg["payback_multiple"] / ratio}, None

    own, why_own = fit(mature[mature.channel == channel], "channel")
    if own:
        return own
    pooled, why_pool = fit(mature, "pooled")
    if pooled:
        return pooled
    return {"level": None, "target_d30_roas": None, "reason": "insufficient_history",
            "channel_history": why_own, "pooled_history": why_pool,
            "needed": {"min_mature_cohorts": pcfg["min_mature_cohorts"], "min_mature_installs": pcfg["min_mature_installs"]}}


def load_mature_history(dec, as_of: str, ann: pd.DataFrame) -> pd.DataFrame:
    """Eligible cohorts (pass the cohort-level quality checks) whose D30 AND D180 milestones are complete and reported at as_of."""
    o30 = cohort_outcomes(dec, as_of, 30)[["channel", "install_date", "installs", "net_revenue"]].rename(columns={"net_revenue": "y30"})
    o180 = cohort_outcomes(dec, as_of, 180)[["channel", "install_date", "net_revenue"]].rename(columns={"net_revenue": "y180"})
    ok = ann[(ann.d3_complete == 1) & ann.exclusion_reason.isna() & ~ann.zero_d3][["channel", "install_date"]]
    return o30.merge(o180, on=["channel", "install_date"]).merge(ok, on=["channel", "install_date"])


# ------------------------------------------------------------------ response curve


def fit_response(obs: pd.DataFrame, rcfg: dict) -> dict:
    """Log-log OLS: log(installs) = a + b*log(spend) + weekday terms + trend, on the most recent decision-ready days.
    obs columns: install_date (ISO text), spend, installs. Returns coefficients, anchor, observed spend range and `usable`
    with explicit reasons when not."""
    d = obs[(obs.spend > 0) & (obs.installs > 0)].sort_values("install_date").tail(rcfg["window_cohort_days"]).copy()
    out = {"n_obs": int(len(d)), "usable": False, "unusable_reasons": []}
    if len(d) < rcfg["min_observations"]:
        out["unusable_reasons"].append(f"only {len(d)} observations, need {rcfg['min_observations']}")
        return out
    dates = pd.to_datetime(d.install_date)
    x = np.log(d.spend.to_numpy(float))
    out["log_spend_sd"] = float(np.std(x, ddof=1))
    weekday = np.stack([(dates.dt.day_of_week == k).to_numpy(float) for k in range(1, 7)], axis=1)
    trend = ((dates - dates.min()).dt.days.to_numpy(float) - 0.5 * (dates.max() - dates.min()).days) / 100.0
    X = np.column_stack([np.ones(len(d)), x, weekday, trend])
    y = np.log(d.installs.to_numpy(float))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(d) - X.shape[1]
    sigma2 = float(resid @ resid / dof)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    b, se = float(beta[1]), float(np.sqrt(cov[1, 1]))
    anchor = d.tail(rcfg["anchor_days"])
    lo_q, hi_q = rcfg["range_quantiles"]
    out.update({"b": b, "b_se": se, "sigma": float(np.sqrt(sigma2)), "i_ref": float(anchor.installs.mean()), "s_ref": float(anchor.spend.mean()),
                "spend_range": [float(np.quantile(d.spend, lo_q)), float(np.quantile(d.spend, hi_q))],
                "window": [str(d.install_date.min()), str(d.install_date.max())]})
    if out["log_spend_sd"] < rcfg["min_log_spend_sd"]:
        out["unusable_reasons"].append(f"spend barely varied (log sd {out['log_spend_sd']:.3f} < {rcfg['min_log_spend_sd']}): elasticity not identified")
    if not (rcfg["b_bounds"][0] <= b <= rcfg["b_bounds"][1]):
        out["unusable_reasons"].append(f"elasticity b = {b:.3f} outside {rcfg['b_bounds']}")
    if se > rcfg["max_b_se"]:
        out["unusable_reasons"].append(f"elasticity standard error {se:.3f} above {rcfg['max_b_se']}")
    out["usable"] = not out["unusable_reasons"]
    return out


def installs_at(resp: dict, spend: float) -> float:
    """Model-estimated daily installs at a daily spend level, anchored at the recent mean spend / installs."""
    return resp["i_ref"] * (spend / resp["s_ref"]) ** resp["b"]


def finite_step(resp: dict, s0: float, s1: float, value_per_install: float, haircut: float, b: float = None, value_scale: float = 1.0) -> dict:
    """Model-estimated finite-step marginal D30 ROAS for raising daily spend from s0 to s1 (per day).
    `b` and `value_scale` let the Monte Carlo reuse this exact formula with drawn values."""
    b = resp["b"] if b is None else b
    i0 = resp["i_ref"] * (s0 / resp["s_ref"]) ** b
    i1 = resp["i_ref"] * (s1 / resp["s_ref"]) ** b
    value = value_per_install * haircut * value_scale
    d_installs, d_rev, d_spend = i1 - i0, (i1 - i0) * value, s1 - s0
    return {"s0": s0, "s1": s1, "installs_at_s0": i0, "installs_at_s1": i1, "extra_installs": d_installs, "extra_revenue": d_rev,
            "extra_spend": d_spend, "marginal_roas": d_rev / d_spend if d_spend > 0 else None,
            # slope of the curve AT s0 (reference only; not what the policy uses)
            "derivative_marginal_roas": value * b * i0 / s0}


def marginal_lower_bound(resp: dict, s0: float, s1: float, value_per_install: float, value_rel_sd: float, mcfg: dict) -> dict:
    """Lower bound of the finite-step marginal ROAS from Monte Carlo draws of the elasticity b ~ N(b_hat, se^2) and the value
    per install ~ N(point, (rel_sd x point)^2). APPROXIMATION: independent normal draws; the forecast interval is read as a
    symmetric normal 80% interval; intercept uncertainty is ignored because the curve is anchored on recent observed values.
    Seeded, so the same inputs always give the same bound. Not a guaranteed 90% bound."""
    rng = np.random.default_rng(mcfg["mc_seed"])
    n = mcfg["mc_draws"]
    b = rng.normal(resp["b"], resp["b_se"], n)
    v = np.clip(rng.normal(1.0, value_rel_sd, n), 0.0, None)
    i0 = resp["i_ref"] * (s0 / resp["s_ref"]) ** b
    i1 = resp["i_ref"] * (s1 / resp["s_ref"]) ** b
    roas = (i1 - i0) * value_per_install * mcfg["incremental_quality_haircut"] * v / (s1 - s0)
    return {"lower": float(np.quantile(roas, mcfg["lower_quantile"])), "median": float(np.median(roas)),
            "upper": float(np.quantile(roas, 1 - mcfg["lower_quantile"])), "draws": int(n), "quantile": mcfg["lower_quantile"]}

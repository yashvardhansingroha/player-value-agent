"""Economics: average vs finite-step marginal ROAS (hand-checkable), payback target, response fit, Monte Carlo bound."""
import numpy as np
import pandas as pd
import pytest

from src.policy import economics as eco
from src.policy.constraints import load_policy

P = load_policy()


def test_average_vs_finite_step_vs_derivative_by_hand():
    """Anchor: 100 installs/day at 100/day spend; elasticity b = 0.5; each install is worth 2.0 (D30 net); haircut 1.0.
    Average ROAS   = (100 x 2.0) / 100                 = 2.000
    Step 100 -> 121: installs 100 x (1.21^0.5) = 110, so +10 installs x 2.0 = +20 revenue for +21 spend -> 20 / 21 = 0.952
    Derivative at 100 (slope at the current budget, NOT the step) = 2.0 x 0.5 x 100 / 100 = 1.000"""
    resp = {"b": 0.5, "i_ref": 100.0, "s_ref": 100.0}
    assert eco.average_roas(100 * 2.0, 100.0) == pytest.approx(2.0)
    fs = eco.finite_step(resp, 100.0, 121.0, value_per_install=2.0, haircut=1.0)
    assert fs["installs_at_s0"] == pytest.approx(100.0) and fs["installs_at_s1"] == pytest.approx(110.0)
    assert fs["extra_installs"] == pytest.approx(10.0) and fs["extra_revenue"] == pytest.approx(20.0) and fs["extra_spend"] == pytest.approx(21.0)
    assert fs["marginal_roas"] == pytest.approx(20 / 21)  # 0.952: below the average of 2.0
    assert fs["derivative_marginal_roas"] == pytest.approx(1.0)  # a different number: the slope at s0
    assert fs["marginal_roas"] < fs["derivative_marginal_roas"] < eco.average_roas(200.0, 100.0)  # concave curve: step < slope < average
    half = eco.finite_step(resp, 100.0, 121.0, 2.0, haircut=0.5)  # incremental players worth half as much
    assert half["marginal_roas"] == pytest.approx(10 / 21)


def test_average_roas_needs_positive_spend():
    assert eco.average_roas(10.0, 0.0) is None and eco.average_roas(10.0, None) is None


def mature(n, y30=100.0, y180=200.0, installs=100, channel="meta"):
    return pd.DataFrame({"channel": channel, "install_date": [f"2025-01-{i + 1:02d}" if i < 28 else f"2025-02-{i - 27:02d}" for i in range(n)],
                         "installs": installs, "y30": y30, "y180": y180})


def test_payback_target_is_payback_multiple_over_d180_to_d30_ratio_of_sums():
    pcfg = dict(P["payback"], min_mature_cohorts=3, min_mature_installs=100)
    df = pd.DataFrame({"channel": "m", "install_date": ["a", "b", "c"], "installs": [100, 100, 100], "y30": [10.0, 20.0, 10.0], "y180": [30.0, 40.0, 30.0]})
    t = eco.payback_target(df, pcfg, "m")
    assert t["level"] == "channel" and t["d180_over_d30"] == pytest.approx(100 / 40)  # ratio of sums 2.5, not mean of ratios
    assert t["target_d30_roas"] == pytest.approx(1.0 / 2.5)  # D30 ROAS needed to repay spend by D180
    assert eco.payback_target(df, dict(pcfg, payback_multiple=1.2), "m")["target_d30_roas"] == pytest.approx(1.2 / 2.5)


def test_payback_target_uses_only_mature_history_and_falls_back_or_says_insufficient():
    pcfg = dict(P["payback"], min_mature_cohorts=30, min_mature_installs=3000)
    own = eco.payback_target(mature(40), pcfg, "meta")
    assert own["level"] == "channel" and own["n_cohorts"] == 40
    pooled = eco.payback_target(pd.concat([mature(10), mature(25, channel="x")]), pcfg, "meta")  # meta has 10 (< 30); pooled has 35
    assert pooled["level"] == "pooled"
    none = eco.payback_target(mature(10), pcfg, "meta")
    assert none["target_d30_roas"] is None and none["reason"] == "insufficient_history" and none["channel_history"]["n_cohorts"] == 10
    trailing = eco.payback_target(mature(120), dict(pcfg, trailing_cohorts=90), "meta")
    assert trailing["n_cohorts"] == 90  # only the most recent 90 mature cohorts
    assert eco.payback_target(mature(40, installs=10), pcfg, "meta")["target_d30_roas"] is None  # too few installs


def synthetic_obs(n=120, b=0.7, noise=0.0, seed=0, constant=False):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-03-03", periods=n)  # starts on a Monday
    spend = np.full(n, 80.0) if constant else 80 * np.exp(0.3 * np.sin(np.arange(n) / 4.0) + 0.1 * np.cos(np.arange(n) / 1.7))
    weekday = np.array([0, -0.02, -0.03, 0, 0.05, 0.12, 0.10])[dates.day_of_week]
    installs = np.exp(np.log(80) + b * np.log(spend / 80) + weekday + 0.0003 * np.arange(n) + rng.normal(0, noise, n))
    return pd.DataFrame({"install_date": dates.strftime("%Y-%m-%d"), "spend": spend, "installs": installs})


def test_response_fit_recovers_a_known_elasticity_with_weekday_and_trend_terms():
    r = eco.fit_response(synthetic_obs(b=0.7), P["response"])
    assert r["usable"] and r["b"] == pytest.approx(0.7, abs=1e-6) and r["b_se"] < 1e-6  # noiseless: exact
    noisy = eco.fit_response(synthetic_obs(b=0.7, noise=0.08, seed=1), P["response"])
    assert noisy["usable"] and abs(noisy["b"] - 0.7) < 0.15 and noisy["b_se"] > 0
    obs = synthetic_obs(b=0.7, noise=0.08, seed=1)
    lo, hi = noisy["spend_range"]
    assert (lo, hi) == pytest.approx((obs.spend.quantile(0.05), obs.spend.quantile(0.95)))  # observed range = 5%..95% spend quantiles


def test_response_fit_is_unusable_with_explicit_reasons():
    few = eco.fit_response(synthetic_obs(n=30), P["response"])
    assert not few["usable"] and "observations" in few["unusable_reasons"][0]
    flat = eco.fit_response(synthetic_obs(constant=True) .assign(installs=lambda d: d.installs * (1 + 0.01 * np.arange(len(d)) % 3)), P["response"])
    assert not flat["usable"] and any("barely varied" in x for x in flat["unusable_reasons"])
    negative = eco.fit_response(synthetic_obs(b=-0.4), P["response"])
    assert not negative["usable"] and any("outside" in x for x in negative["unusable_reasons"])
    assert eco.fit_response(synthetic_obs(n=120, b=0.7, noise=1.5, seed=3), P["response"])["usable"] is False  # too imprecise


def test_monte_carlo_lower_bound_is_deterministic_ordered_and_collapses_without_uncertainty():
    resp = {"b": 0.5, "b_se": 0.1, "i_ref": 100.0, "s_ref": 100.0}
    m = dict(P["marginal"], incremental_quality_haircut=1.0)
    a = eco.marginal_lower_bound(resp, 100.0, 121.0, 2.0, 0.1, m)
    b = eco.marginal_lower_bound(resp, 100.0, 121.0, 2.0, 0.1, m)
    assert a == b  # seeded: same inputs, same bound
    assert a["lower"] < a["median"] < a["upper"] and a["median"] == pytest.approx(20 / 21, abs=0.06)
    exact = eco.marginal_lower_bound(dict(resp, b_se=1e-12), 100.0, 121.0, 2.0, 1e-12, m)
    assert exact["lower"] == pytest.approx(20 / 21, abs=1e-6)  # no uncertainty -> equals the point finite-step estimate
    wide = eco.marginal_lower_bound(dict(resp, b_se=0.3), 100.0, 121.0, 2.0, 0.4, m)
    assert wide["lower"] < a["lower"]  # more uncertainty, lower bound

"""SIMULATED experiment design: units, assignment, the hand-checked estimator, the keep/rollback rule, ONE code path for planning and readout,
and the planning simulation's calibration and uncertainty reporting."""
import numpy as np
import pytest
from scipy import stats as sps

from src.experiments import design as D
from src.experiments import plan as P
from tests.experiment_helpers import CFG

RULE = {"alpha": 0.10, "min_delivery_fraction": 0.5, "max_delivery_fraction": 1.5, "max_cpi_ratio": 1.5, "min_d1_retention_change": -0.03, "target_d30_roas": 1.0}


def test_units_partition_the_budget_exactly_assignment_is_paired_and_the_step_is_allocated_exactly():
    units = D.make_units("meta", 101.7, 24, seed=3)
    assert sum(P.micro(u["baseline_daily_budget"]) for u in units) == P.micro(101.7) and len({u["unit_id"] for u in units}) == 24
    a = D.assign(units, seed=5)
    assert len(a["treated"]) == 12 and a["unpaired_unit"] is None and D.assign(units, seed=5) == a  # deterministic
    by_id = {u["unit_id"]: u["baseline_daily_budget"] for u in units}
    for x, y, t in a["pairs"]:
        assert t in (x, y)
        assert abs(by_id[x] - by_id[y]) <= abs(max(by_id.values()) - min(by_id.values()))  # adjacent in budget order (stratified)
    plan_units = D.allocate_step(units, a["treated"], 10.17)
    assert sum(P.micro(u["planned_incremental_spend"]) for u in plan_units) == P.micro(10.17)
    assert all((u["planned_incremental_spend"] > 0) == (u["arm"] == "treatment") for u in plan_units)
    assert D.assign(units, seed=6)["treated"] != a["treated"]  # a different seed is a different assignment


def test_windows_put_pre_outcomes_before_the_decision_and_readout_after_the_last_cohort_matures():
    w = D.windows("2025-10-13", lag_days=2, pre_days=14, treatment_days=28)
    assert w["pre_end"] == "2025-09-10" and w["pre_start"] == "2025-08-28"  # 30 + 2 + 1 days before the decision date, 14 days long
    assert P.add_days(w["pre_end"], 32) <= "2025-10-13"  # pre-period D30 outcomes are available at the decision date
    assert w["treatment_start"] == "2025-10-13" and w["treatment_end"] == "2025-11-09"
    assert w["readout_available_on"] == "2025-12-11"  # last acquisition day + 30 + 2


def test_exact_allocation_holds_for_very_large_budgets_where_integer_products_would_overflow():
    """Regression: step x baseline in millionths exceeds int64 at a 1,000,000/day channel; the increments must still add up exactly."""
    for baseline, step in ((1_017_000.0, 101_700.0), (10_170_000.0, 1_017_000.123456), (101.7, 10.17)):
        units = D.make_units("meta", baseline, 24, seed=3)
        assert sum(P.micro(u["baseline_daily_budget"]) for u in units) == P.micro(baseline)
        a = D.assign(units, seed=5)
        plan_units = D.allocate_step(units, a["treated"], step)
        treated = [u["planned_incremental_spend"] for u in plan_units if u["arm"] == "treatment"]
        assert sum(P.micro(x) for x in treated) == P.micro(step) and all(x > 0 for x in treated)


def stats_for_hand_check():
    f = lambda *v: np.array([v], dtype=float)  # one replicate; units T1 T2 C1 C2
    return {"R_pre": f(10, 12, 11, 9), "R_post": f(16, 20, 13, 11), "S_pre": f(5, 6, 5, 5), "S_post": f(8, 10, 5, 5),
            "I_pre": f(5, 5, 5, 5), "I_post": f(7, 8, 5, 5), "A_pre": f(2, 2, 2, 2), "A_post": f(3.5, 4, 2, 2)}


def test_estimator_by_hand():
    """Revenue change per unit dR = 6, 8 (treated), 2, 2 (control); spend change dS = 3, 4, 0, 0.
    N = mean_T(dR) - mean_C(dR) = 7 - 2 = 5;  D = mean_T(dS) - mean_C(dS) = 3.5;  iROAS = N / D = 1.428571.
    Residual e = dR - iROAS x dS = 1.714286, 2.285714 (treated), 2, 2 (control): var_T = 0.163265 (n=2), var_C = 0.
    SE = sqrt(0.163265 / 2) / 3.5 = 0.081633; df = 2, t(0.90, 2) = 1.885618; lower bound = 1.428571 - 1.885618 x 0.081633 = 1.274643.
    CPI: treated (8+10)/(7+8) over (5+6)/(5+5) = 1.2 / 1.1; control 1 / 1  ->  ratio 1.090909.   D1: treated 0.4 -> 0.5, control 0.4 -> 0.4: +0.1."""
    est = D.estimate(stats_for_hand_check(), np.array([True, True, False, False]), planned_mean_increment=4.0, alpha=0.10)
    assert est["iroas"][0] == pytest.approx(5 / 3.5)
    assert est["extra_revenue_per_day"][0] == pytest.approx(5.0) and est["extra_spend_per_day"][0] == pytest.approx(3.5)
    assert est["se"][0] == pytest.approx(0.285714 / 3.5, rel=1e-4)
    assert est["lower"][0] == pytest.approx(5 / 3.5 - sps.t.ppf(0.90, 2) * 0.285714 / 3.5, rel=1e-4) and est["lower"][0] == pytest.approx(1.274643, abs=1e-4)
    assert est["delivery_fraction"][0] == pytest.approx(3.5 / 4.0)  # realised extra spend / planned extra spend per treated unit
    assert est["cpi_ratio"][0] == pytest.approx((18 / 15) / (11 / 10), rel=1e-9) and est["d1_retention_change"][0] == pytest.approx(0.1)
    assert (est["n_treatment"], est["n_control"], est["df"]) == (2, 2, 2)


def test_decision_rule_precedence_and_boundaries():
    base = {"iroas": np.array([1.5] * 6), "se": np.array([0.1] * 6), "lower": np.array([1.2, 0.9, 1.2, 1.2, np.nan, 1.0]), "delivery_fraction": np.array([0.9, 0.9, 0.4, 0.9, 0.9, 0.9]),
            "cpi_ratio": np.array([1.0, 1.0, 1.0, 1.6, 1.0, 1.0]), "d1_retention_change": np.array([0.0] * 6)}
    code = D.decide(base, RULE)
    assert list(code) == [0, 1, 3, 2, 4, 0]  # keep; below target; under-delivered; guardrail; NaN lower bound is an INVALID readout; lower == target keeps
    d1 = dict(base, d1_retention_change=np.array([-0.05] * 6))
    assert set(D.decide(d1, RULE)[[0, 5]]) == {2}  # a D1 retention drop trips the guardrail
    over = dict(base, delivery_fraction=np.array([1.6] * 6))
    assert set(D.decide(over, RULE)) == {3}  # over-delivery is also invalid
    assert D.CODES[0] == "keep" and all(D.CODES[c].startswith("rollback") for c in (1, 2, 3, 4))


def one(**over):
    """One replicate whose statistics all pass the rule (keep), with chosen statistics replaced."""
    v = {"iroas": 1.5, "se": 0.1, "lower": 1.2, "delivery_fraction": 0.9, "cpi_ratio": 1.0, "d1_retention_change": 0.0}
    v.update(over)
    return {k: np.array([x], dtype=float) for k, x in v.items()}


def test_decision_fails_closed_on_missing_or_non_finite_statistics_and_the_controls_still_work():
    assert list(D.decide(one(), RULE)) == [0]  # control: everything valid and passing -> keep
    # a guardrail that is missing (NaN) or infinite can never be read as a passing guardrail
    for stat in ("cpi_ratio", "d1_retention_change"):
        for bad in (np.nan, np.inf, -np.inf):
            assert list(D.decide(one(**{stat: bad}), RULE)) == [4], (stat, bad)
    # the same holds for every other statistic the decision needs
    for stat in ("iroas", "se", "lower", "delivery_fraction"):
        for bad in (np.nan, np.inf, -np.inf):
            assert list(D.decide(one(**{stat: bad}), RULE)) != [0], (stat, bad)
    assert list(D.decide(one(iroas=np.nan), RULE)) == [4] and list(D.decide(one(se=np.inf), RULE)) == [4] and list(D.decide(one(lower=-np.inf), RULE)) == [4]
    # impossible values are invalid, not 'passing': a CPI ratio must be positive and a retention change must lie in [-1, 1]
    assert list(D.decide(one(cpi_ratio=0.0), RULE)) == [4] and list(D.decide(one(cpi_ratio=-1.0), RULE)) == [4]
    assert list(D.decide(one(d1_retention_change=1.5), RULE)) == [4] and list(D.decide(one(se=-0.1), RULE)) == [4]
    # controls: valid but failing values keep their own, more specific codes
    assert list(D.decide(one(cpi_ratio=1.6), RULE)) == [2] and list(D.decide(one(d1_retention_change=-0.05), RULE)) == [2]
    assert list(D.decide(one(cpi_ratio=1.5, d1_retention_change=-0.03), RULE)) == [0]  # exactly at the limits passes
    assert list(D.decide(one(lower=0.9), RULE)) == [1] and list(D.decide(one(delivery_fraction=0.4), RULE)) == [3]
    assert list(D.decide(one(delivery_fraction=0.0, iroas=np.inf, lower=np.inf, se=np.inf), RULE)) == [3]  # no extra spend at all: invalid DELIVERY, not just a bad number
    # a statistic that is not supplied at all is a caller error and raises: the decision is never made on a partial estimate
    for stat in D.DECISION_STATISTICS:
        partial = one()
        del partial[stat]
        with pytest.raises(KeyError):
            D.decide(partial, RULE)
    # vectorised: one bad replicate does not contaminate its neighbours
    mixed = {k: np.concatenate([one()[k], one(cpi_ratio=np.nan)[k], one()[k]]) for k in one()}
    assert list(D.decide(mixed, RULE)) == [0, 4, 0]


def test_planning_simulation_counts_invalid_readouts_as_a_rollback_reason():
    cfg = {**CFG, "power": dict(CFG["power"], replicates=50, reported_multiples=[1.0]), "design": dict(CFG["design"], treatment_day_options=[14])}
    out = D.power_study(pre_arrays(), plan_units(), 1.0, {k: RULE[k] for k in RULE if k != "target_d30_roas"}, cfg)["14"]["alternatives"][0]
    assert set(out["rollback_reasons"]) == {D.CODES[c] for c in (1, 2, 3, 4)}  # the invalid-readout reason is reported alongside the others


def test_planning_and_readout_use_the_same_estimator_and_rule(monkeypatch):
    """There is no second implementation: both paths call design.estimate and design.decide."""
    calls = {"estimate": 0, "decide": 0}
    real_e, real_d = D.estimate, D.decide
    monkeypatch.setattr(D, "estimate", lambda *a, **k: (calls.__setitem__("estimate", calls["estimate"] + 1), real_e(*a, **k))[1])
    monkeypatch.setattr(D, "decide", lambda *a, **k: (calls.__setitem__("decide", calls["decide"] + 1), real_d(*a, **k))[1])
    pre, units_plan = pre_arrays(reps=1), plan_units()
    cfg = {**CFG, "power": dict(CFG["power"], replicates=50, reported_multiples=[1.0, 1.5]), "design": dict(CFG["design"], treatment_day_options=[14])}
    D.power_study(pre, units_plan, 1.0, {k: RULE[k] for k in RULE if k != "target_d30_roas"}, cfg)
    assert calls["estimate"] == 2 and calls["decide"] == 2  # planning: once per alternative, each over all replicates at once
    from src.experiments import readout  # the readout's estimator import is the same module attribute
    assert readout.D.estimate is D.estimate and readout.D.decide is D.decide


def pre_arrays(reps=1, units=24, days=14, seed=11, cv=0.5):
    rng = np.random.default_rng(seed)
    base = rng.uniform(4, 8, units)
    spend = base[:, None] * rng.lognormal(-0.01, 0.1, (units, days))
    installs = (4 * base[:, None] / 5) * rng.lognormal(-0.1, 0.45, (units, days))
    value = 1.2
    return {"spend": spend, "installs": installs, "d1_active": installs * 0.4 * rng.lognormal(-0.02, 0.2, (units, days)),
            "d30_net_revenue": installs * value * rng.lognormal(-0.5 * cv**2, cv, (units, days))}


def plan_units(units=24, step=10.0, seed=1):
    u = D.make_units("x", 100.0, units, seed)
    a = D.assign(u, seed)
    return D.allocate_step(u, a["treated"], step)


def test_planning_simulation_is_deterministic_reports_replicates_and_uncertainty_and_increases_with_the_true_effect():
    cfg = {**CFG, "power": dict(CFG["power"], replicates=400, reported_multiples=[0.5, 1.0, 1.5, 3.0]), "design": dict(CFG["design"], treatment_day_options=[14, 28])}
    pre, pu = pre_arrays(), plan_units()
    rb = {k: RULE[k] for k in RULE if k != "target_d30_roas"}
    a, b = D.power_study(pre, pu, 1.0, rb, cfg), D.power_study(pre, pu, 1.0, rb, cfg)
    assert a == b  # deterministic: seeds derive from the declared seed, duration and alternative
    for dur in ("14", "28"):
        alts = a[dur]["alternatives"]
        assert a[dur]["replicates"] == 400 and [x["multiple"] for x in alts] == [0.5, 1.0, 1.5, 3.0]
        rates = [x["keep_rate"] for x in alts]
        assert rates == sorted(rates)  # P(keep) rises with the true iROAS
        for x in alts:
            assert x["mc_se"] == pytest.approx(np.sqrt(x["keep_rate"] * (1 - x["keep_rate"]) / 400), abs=1e-3) or x["mc_se"] < 0.001
    assert a["28"]["alternatives"][2]["keep_rate"] >= a["14"]["alternatives"][2]["keep_rate"] - 0.1  # longer windows do not lose power (noise allowance)


def test_false_keep_at_the_boundary_is_near_the_nominal_level_when_noise_is_benign():
    """With light-tailed noise the one-sided 90% lower bound should keep about 10% of the time when the truth equals the target."""
    cfg = {**CFG, "power": dict(CFG["power"], replicates=3000, reported_multiples=[0.5, 1.0]), "design": dict(CFG["design"], treatment_day_options=[28])}
    out = D.power_study(pre_arrays(cv=0.15), plan_units(step=40.0), 1.0, {k: RULE[k] for k in RULE if k != "target_d30_roas"}, cfg)["28"]["alternatives"]
    boundary, below = out[1], out[0]
    assert abs(boundary["keep_rate"] - 0.10) < 0.045  # 3000 replicates: MC SE about 0.005; the t approximation is not exact
    assert below["keep_rate"] < boundary["keep_rate"]


def test_adequacy_selection_picks_the_shortest_adequate_window_or_none():
    def block(days, power, fk):
        alts = [{"multiple": m, "true_iroas": m, "keep_rate": r, "mc_se": 0.01, "median_estimate": m, "median_lower_bound": 0.0, "median_se": 0.1, "rollback_reasons": {}}
                for m, r in ((0.5, 0.0), (0.8, 0.0), (1.0, fk), (1.25, 0.5), (1.5, power), (2.0, 0.99))]
        return {"treatment_days": days, "replicates": 1000, "alternatives": alts}
    cfg = {**CFG, "design": dict(CFG["design"], treatment_day_options=[14, 28, 56])}
    pa = D.assess_design({"14": block(14, 0.3, 0.05), "28": block(28, 0.85, 0.05), "56": block(56, 0.95, 0.05)}, cfg)
    assert pa["chosen_treatment_days"] == 28 and pa["adequate"] is True  # 14 lacks power; the shortest adequate is 28
    none = D.assess_design({"14": block(14, 0.3, 0.05), "28": block(28, 0.5, 0.05), "56": block(56, 0.95, 0.30)}, cfg)
    assert none["chosen_treatment_days"] is None and none["adequate"] is False and none["reasons"]  # 56 has power but leaks false keeps
    assert none["per_duration"]["56"]["inadequacy_reasons"][0].startswith("false-keep")

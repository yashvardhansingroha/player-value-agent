"""SIMULATED experiment design: units, assignment, the estimator and keep/rollback rule, the planning simulation and the power study.
ALL DATA IS SYNTHETIC. Nothing here reads hidden truth or any outcome of the experiment being planned.

ONE CODE PATH: `estimate()` and `decide()` are vectorised over a leading "replicate" axis. The planning simulation calls them with
1000 replicates; the actual readout calls them with 1. There is no second implementation of the rule.

DESIGN IN PLAIN LANGUAGE
  * The channel's daily budget is split into 24 market units (their budgets add up to the channel budget exactly).
  * Units are paired by pre-period budget and one of each pair is randomly given the increase; the OTHER units keep their budget.
    The whole approved channel increase is spent in the treated units, so the treated units' budgets add up to exactly that step.
  * For each unit we compare its average daily D30 net revenue and spend AFTER the change with BEFORE, then compare treated with
    control units (difference-in-differences). iROAS = extra revenue / extra spend. Its uncertainty comes from the spread across units.
  * KEEP only if the one-sided lower bound of iROAS reaches the target D30 ROAS and the guardrails and spend delivery are fine;
    otherwise ROLL BACK.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats as sps

from src.experiments import plan as P
from src.ledger.proposal import canonical

ROOT = Path(__file__).resolve().parents[2]
UNIT_SHARE_SIGMA = 0.5  # spread of unit budget shares (the simulator reads each unit's share from its observable baseline budget)
CODES = {0: "keep", 1: "rollback:lower_bound_below_target", 2: "rollback:guardrail_breached", 3: "rollback:invalid_spend_delivery",
         4: "rollback:invalid_readout_statistic"}
# every statistic the decision needs: if any is missing (NaN) or infinite the readout is INVALID and the answer is rollback, never keep
DECISION_STATISTICS = ("iroas", "se", "lower", "delivery_fraction", "cpi_ratio", "d1_retention_change")


def load_config(path=None) -> dict:
    return P.load_experiment_config(path)


def config_hash(cfg: dict) -> str:
    return P.config_hash(cfg)


def derive_seed(*parts) -> int:
    """Deterministic 32-bit seed from the parts (never chosen by hand after seeing a result)."""
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------- units and assignment


def make_units(channel: str, baseline_budget: float, k: int, seed: int) -> list:
    """k units that PARTITION the baseline budget exactly (in millionths): lognormal shares, remainder to the largest unit."""
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0.0, UNIT_SHARE_SIGMA, k)
    w /= w.sum()
    total = P.micro(baseline_budget)
    m = np.floor(w * total).astype(np.int64)
    m[np.argmax(m)] += total - m.sum()
    return [{"unit_id": f"{channel}-u{i + 1:02d}", "baseline_daily_budget": int(x) / 1e6} for i, x in enumerate(m)]


def assign(units: list, seed: int) -> dict:
    """Stratified randomisation: sort units by baseline budget, pair neighbours, one of each pair is treated (coin flip per pair)."""
    rng = np.random.default_rng(seed)
    order = sorted(units, key=lambda u: (u["baseline_daily_budget"], u["unit_id"]))
    pairs, treated = [], []
    for a, b in zip(order[0::2], order[1::2]):
        flip = int(rng.integers(0, 2))
        t = (a, b)[flip]["unit_id"]
        pairs.append([a["unit_id"], b["unit_id"], t])
        treated.append(t)
    leftover = order[-1]["unit_id"] if len(order) % 2 else None
    return {"method": "stratified by baseline budget: adjacent pairs, one treated per pair by seeded coin flip", "seed": int(seed),
            "pairs": pairs, "unpaired_unit": leftover, "treated": sorted(treated)}


def allocate_step(units: list, treated_ids: list, step_amount: float) -> list:
    """Spread the channel step over the treated units in proportion to their baseline budgets, exactly (millionths, remainder to the
    largest treated unit), so the increments add up to the approved step with no float error."""
    step = P.micro(step_amount)
    tr = [u for u in units if u["unit_id"] in set(treated_ids)]
    base = np.array([P.micro(u["baseline_daily_budget"]) for u in tr], dtype=np.int64)
    inc = np.floor(step * (base / base.sum())).astype(np.int64)  # proportions in float: step * base can exceed int64 for large budgets
    inc[np.argmax(base)] += step - inc.sum()  # exact integer remainder, so the increments add up to the step to the millionth
    by_id = {u["unit_id"]: int(i) / 1e6 for u, i in zip(tr, inc)}
    return [{"unit_id": u["unit_id"], "arm": "treatment" if u["unit_id"] in by_id else "control", "baseline_daily_budget": u["baseline_daily_budget"],
             "planned_incremental_spend": by_id.get(u["unit_id"], 0.0)} for u in units]


def windows(decision_as_of: str, lag_days: int, pre_days: int, treatment_days: int, horizon: int = P.SUPPORTED_HORIZON_DAYS) -> dict:
    """Pre-period ends early enough that its D30 outcomes are available at the decision date; treatment starts on the decision date."""
    pre_end = P.add_days(decision_as_of, -(horizon + lag_days + 1))
    return {"pre_start": P.add_days(pre_end, -(pre_days - 1)), "pre_end": pre_end, "treatment_start": decision_as_of,
            "treatment_end": P.add_days(decision_as_of, treatment_days - 1), "pre_days": pre_days, "treatment_days": treatment_days,
            "outcome_horizon_days": horizon, "reporting_lag_days": lag_days,
            "readout_available_on": P.add_days(P.add_days(decision_as_of, treatment_days - 1), horizon + lag_days)}


# ---------------------------------------------------------------- the estimator and the keep/rollback rule (vectorised)


def unit_stats(spend, installs, d1, rev, n_pre: int) -> dict:
    """Per-unit period means from daily arrays shaped (..., units, days); the first n_pre days are the pre-period."""
    pre, post = slice(0, n_pre), slice(n_pre, None)
    return {"S_pre": spend[..., pre].mean(-1), "S_post": spend[..., post].mean(-1), "I_pre": installs[..., pre].mean(-1), "I_post": installs[..., post].mean(-1),
            "A_pre": d1[..., pre].mean(-1), "A_post": d1[..., post].mean(-1), "R_pre": rev[..., pre].mean(-1), "R_post": rev[..., post].mean(-1)}


def estimate(st: dict, treat: np.ndarray, planned_mean_increment: float, alpha: float) -> dict:
    """Unit-level difference-in-differences ratio with a delta-method standard error and a one-sided t lower bound.
    `st` arrays are (replicates, units); `treat` is a boolean mask over units. Returns arrays shaped (replicates,)."""
    T, C = treat, ~treat
    nT, nC = int(T.sum()), int(C.sum())
    dR, dS = st["R_post"] - st["R_pre"], st["S_post"] - st["S_pre"]
    N = dR[:, T].mean(1) - dR[:, C].mean(1)  # extra revenue per day (treated minus control change)
    D = dS[:, T].mean(1) - dS[:, C].mean(1)  # extra spend per day (realised)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = N / D
        e = dR - r[:, None] * dS  # residual of the unit revenue change after removing r x its spend change
        var = e[:, T].var(1, ddof=1) / nT + e[:, C].var(1, ddof=1) / nC
        se = np.sqrt(var) / np.abs(D)
        lower = r - sps.t.ppf(1 - alpha, nT + nC - 2) * se
        delivery = D / planned_mean_increment
        cpi = lambda S, I: S.sum(1) / I.sum(1)
        cpi_ratio = (cpi(st["S_post"][:, T], st["I_post"][:, T]) / cpi(st["S_pre"][:, T], st["I_pre"][:, T])) / \
                    (cpi(st["S_post"][:, C], st["I_post"][:, C]) / cpi(st["S_pre"][:, C], st["I_pre"][:, C]))
        ret = lambda A, I: A.sum(1) / I.sum(1)
        d1_change = (ret(st["A_post"][:, T], st["I_post"][:, T]) - ret(st["A_pre"][:, T], st["I_pre"][:, T])) - \
                    (ret(st["A_post"][:, C], st["I_post"][:, C]) - ret(st["A_pre"][:, C], st["I_pre"][:, C]))
    return {"iroas": r, "se": se, "lower": lower, "extra_revenue_per_day": N, "extra_spend_per_day": D, "delivery_fraction": delivery,
            "cpi_ratio": cpi_ratio, "d1_retention_change": d1_change, "n_treatment": nT, "n_control": nC, "df": nT + nC - 2}


def decide(est: dict, rule: dict) -> np.ndarray:
    """Integer code per replicate. FAILS CLOSED: only code 0 means keep.
        3  spend delivery is a finite number outside the allowed band (including no extra spend at all)
        4  any statistic the decision needs is missing (NaN), infinite or impossible: iROAS, SE, lower bound, delivery, the CPI ratio
           or the D1 retention change (a CPI ratio must be > 0, a retention change must lie in [-1, 1]). The readout is INVALID.
        2  a guardrail is breached
        1  the lower bound is below the target
        0  keep: every statistic is valid, delivery is in the band, both guardrails pass, and lower bound >= target
    Precedence is the order above. A missing guardrail can therefore never be read as a passing one."""
    stat = {k: np.asarray(est[k], dtype=float) for k in DECISION_STATISTICS}
    d = stat["delivery_fraction"]
    with np.errstate(invalid="ignore"):
        bad_delivery = np.isfinite(d) & ((d < rule["min_delivery_fraction"]) | (d > rule["max_delivery_fraction"]))
        usable = np.ones(d.shape, dtype=bool)
        for v in stat.values():
            usable &= np.isfinite(v)
        usable &= (stat["cpi_ratio"] > 0) & (stat["se"] >= 0) & (np.abs(stat["d1_retention_change"]) <= 1.0)
        guard = (stat["cpi_ratio"] > rule["max_cpi_ratio"]) | (stat["d1_retention_change"] < rule["min_d1_retention_change"])
        below = stat["lower"] < rule["target_d30_roas"]
    return np.where(bad_delivery, 3, np.where(~usable, 4, np.where(guard, 2, np.where(below, 1, 0))))


def rule_from_plan(plan: dict) -> dict:
    """The decision rule exactly as frozen in the plan (the readout never reads thresholds from anywhere else)."""
    est, g, dr = plan["estimator"], plan["guardrails"], plan["decision_rule"]
    return {"alpha": est["alpha"], "min_delivery_fraction": est["min_delivery_fraction"], "max_delivery_fraction": est["max_delivery_fraction"],
            "max_cpi_ratio": g["max_cpi_ratio"], "min_d1_retention_change": g["min_d1_retention_change"], "target_d30_roas": dr["target_d30_roas"]}


def treat_mask(plan: dict, unit_order: list) -> np.ndarray:
    arm = {u["unit_id"]: u["arm"] for u in plan["units"]}
    return np.array([arm[u] == "treatment" for u in unit_order])


def planned_mean_increment(plan: dict) -> float:
    inc = [u["planned_incremental_spend"] for u in plan["units"] if u["arm"] == "treatment"]
    return float(sum(inc) / len(inc))


def aggregate_observations(obs: pd.DataFrame, plan: dict) -> tuple:
    """Daily observation rows (unit_id, acquisition_date, spend, installs, d1_active, d30_net_revenue) -> (stats dict with a
    leading replicate axis of 1, unit order, input problems). Requires exactly the plan's pre and treatment days for every unit.

    A missing value (NaN/None) is UNAVAILABLE, which is not the same as a valid zero: zero spend, zero installs or zero revenue on a
    day are real observations and are kept as 0. Cells that are unavailable (NaN/inf) or impossible (negative spend, installs or
    revenue; D1-active players above installs) are set to NaN, so every statistic that depends on them is NaN and `decide` returns
    the invalid-readout rollback. The counts are returned so the readout can say what was wrong."""
    w = plan["windows"]
    days = pd.date_range(w["pre_start"], w["pre_end"]).strftime("%Y-%m-%d").tolist() + pd.date_range(w["treatment_start"], w["treatment_end"]).strftime("%Y-%m-%d").tolist()
    units = [u["unit_id"] for u in plan["units"]]
    obs = obs[obs.unit_id.isin(units) & obs.acquisition_date.isin(days)]
    if len(obs) != len(units) * len(days) or obs.duplicated(["unit_id", "acquisition_date"]).any():
        raise ValueError(f"expected {len(units) * len(days)} unit-day rows, found {len(obs)}")
    grid = {c: obs.pivot(index="unit_id", columns="acquisition_date", values=c).loc[units, days].to_numpy(float)[None]
            for c in ("spend", "installs", "d1_active", "d30_net_revenue")}
    problems = {c: int((~np.isfinite(g)).sum()) for c, g in grid.items()}  # unavailable cells per column
    with np.errstate(invalid="ignore"):
        impossible = {c: (grid[c] < 0) for c in grid}
        impossible["d1_active"] = impossible["d1_active"] | (grid["d1_active"] > grid["installs"])
    for c in grid:
        problems[c] += int(impossible[c].sum())
        grid[c] = np.where(impossible[c], np.nan, np.where(np.isfinite(grid[c]), grid[c], np.nan))
    return unit_stats(grid["spend"], grid["installs"], grid["d1_active"], grid["d30_net_revenue"], w["pre_days"]), units, {c: n for c, n in problems.items() if n}


# ---------------------------------------------------------------- planning simulation (bootstrap from PRE-period observations only)


def planning_model(pre: dict) -> dict:
    """pre: arrays (units, pre_days) of spend, installs, d1_active, d30_net_revenue observed BEFORE the plan. Noise = bootstrap of
    unit-day rows normalised by each unit's own mean (rows are resampled whole, so their correlation is kept)."""
    m = {k: pre[k].mean(1) for k in ("spend", "installs", "d1_active", "d30_net_revenue")}
    if any((v <= 0).any() for v in m.values()):
        raise ValueError("a unit has a zero pre-period mean; the planning bootstrap cannot be built")
    pool = np.stack([pre[k] / m[k][:, None] for k in ("spend", "installs", "d1_active", "d30_net_revenue")], axis=-1).reshape(-1, 4)
    return {"means": m, "pool": pool, "n_units": pre["spend"].shape[0], "n_pre": pre["spend"].shape[1]}


def simulate_planning(model: dict, increments: np.ndarray, treat: np.ndarray, n_post: int, alt_iroas: float, reps: int, rng, delivery: float, shock_sd: float) -> dict:
    """Replicates of the experiment under a declared TRUE iROAS (`alt_iroas`, per unit of REALISED extra spend). Returns unit_stats arrays."""
    U, days = model["n_units"], model["n_pre"] + n_post
    draw = model["pool"][rng.integers(0, len(model["pool"]), (reps, U, days))]  # (reps, U, days, 4)
    mu = model["means"]
    spend = mu["spend"][None, :, None] * draw[..., 0]
    installs = mu["installs"][None, :, None] * draw[..., 1]
    d1 = mu["d1_active"][None, :, None] * draw[..., 2]
    rev = mu["d30_net_revenue"][None, :, None] * draw[..., 3]
    post = slice(model["n_pre"], None)
    shock = np.exp(rng.normal(0.0, shock_sd, (reps, 1, 1)))  # a period-wide shock shared by both arms
    installs[..., post] *= shock
    d1[..., post] *= shock
    rev[..., post] *= shock
    inc = np.where(treat, increments, 0.0)[None, :, None]  # planned extra spend per treated unit per day
    inc_rev = alt_iroas * delivery * inc
    value = (mu["d30_net_revenue"] / mu["installs"])[None, :, None]
    inc_inst = inc_rev / value
    retention = (mu["d1_active"] / mu["installs"])[None, :, None]
    spend[..., post] += inc * delivery * draw[..., post, 0]
    rev[..., post] += inc_rev
    installs[..., post] += inc_inst
    d1[..., post] += inc_inst * retention
    return unit_stats(spend, installs, d1, rev, model["n_pre"])


def power_study(pre: dict, units_plan: list, target: float, rule_base: dict, cfg: dict, extra_multiples=()) -> dict:
    """Power and false-keep for every allowed treatment window, using the SAME estimate()/decide() as the readout."""
    pw, de = cfg["power"], cfg["design"]
    unit_ids = [u["unit_id"] for u in units_plan]
    treat = np.array([u["arm"] == "treatment" for u in units_plan])
    inc = np.array([u["planned_incremental_spend"] for u in units_plan])
    model = planning_model(pre)
    rule = dict(rule_base, target_d30_roas=target)
    mean_inc = float(inc[treat].mean())
    multiples = sorted(set(pw["reported_multiples"]) | set(extra_multiples))
    out = {}
    for n_post in de["treatment_day_options"]:
        alts = []
        for m in multiples:
            rng = np.random.default_rng(derive_seed(pw["seed"], n_post, m))
            st = simulate_planning(model, inc, treat, n_post, m * target, pw["replicates"], rng, de["planning_delivery_fraction"], de["planning_common_shock_sd"])
            est = estimate(st, treat, mean_inc, rule["alpha"])
            code = decide(est, rule)
            p = float((code == 0).mean())
            alts.append({"multiple": m, "true_iroas": m * target, "keep_rate": p, "mc_se": float(np.sqrt(max(p * (1 - p), 1e-12) / pw["replicates"])),
                         "median_estimate": float(np.nanmedian(est["iroas"])), "median_lower_bound": float(np.nanmedian(est["lower"])),
                         "median_se": float(np.nanmedian(est["se"])),
                         "rollback_reasons": {CODES[c]: float((code == c).mean()) for c in (1, 2, 3, 4)}})
        out[str(n_post)] = {"treatment_days": n_post, "replicates": pw["replicates"], "alternatives": alts}
    return out


def standards(cfg: dict) -> dict:
    return P.standards(cfg)


def assess_design(per_duration: dict, cfg: dict) -> dict:
    """Pick the shortest allowed treatment window whose recorded numbers meet the standards; otherwise none (inadequate)."""
    std = standards(cfg)
    for n_post in cfg["design"]["treatment_day_options"]:
        trial = {"standards": std, "chosen_treatment_days": n_post, "per_duration": per_duration}
        ok, reasons = P.adequacy_from_numbers(trial)
        per_duration[str(n_post)]["adequate"], per_duration[str(n_post)]["inadequacy_reasons"] = ok, reasons
    chosen = next((n for n in cfg["design"]["treatment_day_options"] if per_duration[str(n)]["adequate"]), None)
    pa = {"standards": std, "replicates": cfg["power"]["replicates"], "seed": cfg["power"]["seed"], "per_duration": per_duration, "chosen_treatment_days": chosen}
    pa["adequate"], pa["reasons"] = P.adequacy_from_numbers(pa)
    pa["note"] = ("Planning simulation: bootstrap of PRE-period unit-day noise with a declared true iROAS; same estimator and decision rule as the readout. "
                  "keep_rate at 1.0x target and below is the false-keep rate; keep_rate at the declared alternative is power; mc_se is the Monte Carlo standard error.")
    return pa


def build_plan(*, channel: str, decision_as_of: str, policy_hash: str, baseline_budget: float, proposed_budget: float, target: dict, model_estimate: dict,
               units: list, pre_obs: pd.DataFrame, lag_days: int, cfg: dict, scale_note: str = None) -> dict:
    """Assemble, power-check and return a (not yet registered) plan. If no allowed treatment window is adequate the plan says so
    (`power_assessment.adequate` False) and the policy action for the channel must become insufficient_evidence."""
    de = cfg["design"]
    step = round(proposed_budget - baseline_budget, 6)
    seed = derive_seed(de["assignment_seed_base"], channel, decision_as_of)
    assignment = assign(units, seed)
    plan_units = allocate_step(units, assignment["treated"], step)
    w0 = windows(decision_as_of, lag_days, de["pre_days"], de["treatment_day_options"][-1])
    unit_ids = [u["unit_id"] for u in plan_units]
    pre_days = pd.date_range(w0["pre_start"], w0["pre_end"]).strftime("%Y-%m-%d").tolist()
    pre = {}
    for col in ("spend", "installs", "d1_active", "d30_net_revenue"):
        grid = pre_obs[pre_obs.acquisition_date.isin(pre_days)].pivot(index="unit_id", columns="acquisition_date", values=col)
        pre[col] = grid.loc[unit_ids, pre_days].to_numpy(float)
    rule_base = {"alpha": cfg["estimator"]["alpha"], "min_delivery_fraction": cfg["estimator"]["min_delivery_fraction"],
                 "max_delivery_fraction": cfg["estimator"]["max_delivery_fraction"], "max_cpi_ratio": cfg["guardrails"]["max_cpi_ratio"],
                 "min_d1_retention_change": cfg["guardrails"]["min_d1_retention_change"]}
    per_duration = power_study(pre, plan_units, target["target_d30_roas"], rule_base, cfg,
                               extra_multiples=[round(model_estimate["finite_step_marginal_roas"] * cfg["power"]["informational_planning_haircut"] / target["target_d30_roas"], 6)])
    pa = assess_design(per_duration, cfg)
    chosen = pa["chosen_treatment_days"] or de["treatment_day_options"][-1]
    plan = {
        "schema": P.SCHEMA, "simulated": True, "label": P.LABEL, "channel": channel, "decision_as_of": decision_as_of,
        "policy_hash": policy_hash, "experiment_config_hash": config_hash(cfg),
        "budget_step": {"baseline_daily_budget": round(baseline_budget, 6), "proposed_daily_budget": round(proposed_budget, 6), "step_amount": step,
                        "step_pct": round(step / baseline_budget, 6)},
        "target": target, "model_estimate": model_estimate, "units": plan_units, "assignment": assignment,
        "windows": windows(decision_as_of, lag_days, de["pre_days"], chosen),
        "estimator": P.expected_estimator(cfg),
        "guardrails": dict(cfg["guardrails"]),
        "decision_rule": {"keep_if": "spend delivery within bounds AND guardrails pass AND lower bound of iROAS >= target_d30_roas; otherwise rollback",
                          "target_d30_roas": target["target_d30_roas"], "alpha": cfg["estimator"]["alpha"]},
        "power_assessment": pa, "status": "frozen_before_outcomes",
    }
    pa["design_fingerprint"] = P.design_fingerprint(plan)  # ties the recorded numbers to THIS design (units, arms, step, target, thresholds, pre-period)
    if scale_note:
        plan["scale_note"] = scale_note
    return plan

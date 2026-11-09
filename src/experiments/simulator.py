"""HIDDEN-SIDE simulator of the unit-level experiment outcomes. SIMULATED. ALL DATA IS SYNTHETIC.

This is the ONE experiment module allowed to read hidden truth (the channel's true response curve and the planted regime shift) and
the simulator parameters (config/experiment_sim.yaml). It turns a plan's frozen budgets into NOISY observations (spend, installs,
D1-active installs, D30 net revenue per unit and acquisition day). The readout and the planning simulation never import this file;
the readout sees only the observation rows that were written to the experiment store.

Truth model (the simulator's OWN generative model, not the planning bootstrap):
  * unit u holds a fixed share w_u of the channel budget and behaves like a miniature of the channel: expected daily installs
    = rho_u * w_u * scale * F(spend_u / (w_u * scale)), with F the channel's TRUE spend->installs curve and rho_u a unit multiplier;
  * installs are negative-binomial around that mean and share a day-wide shock;
  * value per install falls as the unit's spend rises above its baseline: v = v_ref * nu_u * (spend / baseline)^(-gamma) * regime(day),
    where gamma is the channel's true player-quality exponent and regime is the planted monetisation shift;
  * D30 revenue = ad revenue + a heavy-tailed sum over paying installs; D1 retention falls with the same quality dilution;
  * treated units receive only `delivery` (about 97%) of their planned extra spend.
`scale` multiplies budgets and install volume (the 'what volume would this test need?' study); 1.0 is the dev-seed channel.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from src.data.metrics import cohort_outcomes
from src.experiments import design as D
from src.experiments import plan as P

ROOT = Path(__file__).resolve().parents[2]


def load_sim_config(path=None) -> dict:
    with open(path or ROOT / "config" / "experiment_sim.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_hidden(seed: int = 1, root=None) -> dict:
    return json.loads((Path(root or ROOT / "data" / "hidden") / f"seed_{seed}" / "hidden_truth.json").read_text(encoding="utf-8"))


def value_reference(dec, channel: str, as_of: str, n: int) -> float:
    """Mean D30 net revenue per install of the channel's last `n` cohorts that were matured at the decision date."""
    o = cohort_outcomes(dec, as_of, 30)
    o = o[o.channel == channel].sort_values("install_date").tail(n)
    return float(o.net_revenue.sum() / o.installs.sum())


class ExperimentSimulator:
    def __init__(self, truth: dict, channel: str, unit_budgets: dict, v_ref: float, sim_cfg: dict, scale: float = 1.0, unit_seed: int = 0):
        ch = truth["channels"][channel]
        self.resp, self.gamma, self.v_ref, self.cfg, self.scale = ch["response"], ch["quality_gamma"], v_ref, sim_cfg, scale
        self.r1 = truth["resolved_config"]["channels"][channel]["retention"]["r1"]
        self.regime, self.start_date = truth["planted"]["regime_shift"], truth["resolved_config"]["start_date"]
        self.unit_ids = sorted(unit_budgets)
        self.b = np.array([unit_budgets[u] for u in self.unit_ids])
        self.w = self.b / self.b.sum()
        rng = np.random.default_rng(unit_seed)
        u = sim_cfg["units"]
        self.rho = rng.lognormal(-0.5 * u["installs_multiplier_sigma"] ** 2, u["installs_multiplier_sigma"], len(self.b))
        self.nu = rng.lognormal(-0.5 * u["value_multiplier_sigma"] ** 2, u["value_multiplier_sigma"], len(self.b))

    # ---- truth functions (noiseless expectations)
    def F(self, s):
        r = self.resp
        if r["type"] == "power":
            return r["a"] * np.power(s, r["b"])
        return r["imax"] * np.power(s, r["h"]) / (r["k"] ** r["h"] + np.power(s, r["h"]))

    def regime_factor(self, iso_date: str) -> float:
        day = (date.fromisoformat(iso_date) - date.fromisoformat(self.start_date)).days
        if day < self.regime["start_day"]:
            return 1.0
        share = self.cfg["revenue"]["iap_share_of_value"]
        return self.regime["iap_purchase_probability_multiplier"] * share + self.regime["ecpm_multiplier"] * (1 - share)

    def expected_installs(self, spend):
        return self.rho * self.w * self.scale * self.F(spend / (self.w * self.scale))

    def expected_value(self, spend, iso_date: str):
        return self.v_ref * self.nu * np.power(spend / self.b, -self.gamma) * self.regime_factor(iso_date)

    # ---- one acquisition day of noisy observations for every unit
    def day(self, iso_date: str, planned: np.ndarray, treated_increment: np.ndarray, seed_root: int) -> dict:
        """planned: baseline daily budgets (units,); treated_increment: extra planned spend for treated units on this day (0 elsewhere)."""
        rng = np.random.default_rng(np.random.SeedSequence([seed_root, date.fromisoformat(iso_date).toordinal()]))
        nz, rv, U = self.cfg["noise"], self.cfg["revenue"], len(planned)
        delivery = np.clip(rng.normal(nz["treated_delivery_mean"], nz["treated_delivery_sd"], U), 0.0, None)
        spend = (planned + treated_increment * delivery) * rng.lognormal(-0.5 * nz["spend_noise_sd"] ** 2, nz["spend_noise_sd"], U)
        common = rng.lognormal(-0.5 * nz["common_day_sd"] ** 2, nz["common_day_sd"])
        phi = nz["installs_dispersion"]
        installs = rng.poisson(self.expected_installs(spend) * common * rng.gamma(1 / phi, phi, U))
        q = np.power(spend / self.b, -self.gamma)
        d1 = rng.binomial(installs, np.clip(self.r1 * q, 0.0, 1.0))
        v = self.expected_value(spend, iso_date)
        ad = (1 - rv["iap_share_of_value"]) * v * installs * np.clip(1 + rv["ad_noise_sd"] / np.sqrt(np.maximum(installs, 1)) * rng.normal(size=U), 0.0, None)
        payers = rng.binomial(installs, rv["payer_rate"])
        mean_pay = rv["iap_share_of_value"] * v / rv["payer_rate"]
        mu = np.log(mean_pay) - 0.5 * rv["payer_sigma"] ** 2
        draws = rng.lognormal(np.repeat(mu, payers), rv["payer_sigma"])
        iap = np.bincount(np.repeat(np.arange(U), payers), weights=draws, minlength=U)
        return {"spend": spend, "installs": installs, "d1_active": d1, "d30_net_revenue": ad + iap}

    # ---- rows for the experiment store
    def rows(self, iso_date: str, phase: str, planned: np.ndarray, increment: np.ndarray, seed_root: int, lag: int, plan_hash=None) -> list:
        d = self.day(iso_date, planned, increment, seed_root)
        avail = P.add_days(iso_date, 30 + lag)
        return [{"unit_id": u, "acquisition_date": iso_date, "phase": phase, "spend": float(d["spend"][i]), "installs": int(d["installs"][i]),
                 "d1_active": int(d["d1_active"][i]), "d30_net_revenue": float(d["d30_net_revenue"][i]), "available_on": avail, "plan_hash": plan_hash}
                for i, u in enumerate(self.unit_ids)]

    def pre_rows(self, window: dict, seed_root: int) -> list:
        out = []
        for iso in np.datetime_as_string(np.arange(window["pre_start"], P.add_days(window["pre_end"], 1), dtype="datetime64[D]")):
            out += self.rows(str(iso), "pre", self.b, np.zeros_like(self.b), seed_root, window["reporting_lag_days"])
        return out

    def treatment_rows_up_to(self, plan: dict, plan_hash: str, clock_date: str, seed_root: int, already: set) -> list:
        """Rows for treatment acquisition days whose outcome is available at `clock_date` and not yet stored."""
        w = plan["windows"]
        inc = {u["unit_id"]: u["planned_incremental_spend"] for u in plan["units"]}
        increment = np.array([inc[u] for u in self.unit_ids])
        out = []
        for iso in np.datetime_as_string(np.arange(w["treatment_start"], P.add_days(w["treatment_end"], 1), dtype="datetime64[D]")):
            iso = str(iso)
            if iso in already or P.add_days(iso, 30 + w["reporting_lag_days"]) > clock_date:
                continue
            out += self.rows(iso, "treatment", self.b, increment, seed_root, w["reporting_lag_days"], plan_hash)
        return out

    def arrays(self, plan: dict, seed_root: int) -> dict:
        """Whole experiment in memory (for the evaluation-side replicate study): daily arrays (units, days), pre then treatment."""
        w = plan["windows"]
        inc = {u["unit_id"]: u["planned_incremental_spend"] for u in plan["units"]}
        increment = np.array([inc[u] for u in self.unit_ids])
        cols = {k: [] for k in ("spend", "installs", "d1_active", "d30_net_revenue")}
        for phase, a, b in (("pre", w["pre_start"], w["pre_end"]), ("treatment", w["treatment_start"], w["treatment_end"])):
            for iso in np.datetime_as_string(np.arange(a, P.add_days(b, 1), dtype="datetime64[D]")):
                d = self.day(str(iso), self.b, increment if phase == "treatment" else np.zeros_like(self.b), seed_root)
                for k in cols:
                    cols[k].append(d[k])
        return {k: np.stack(v, axis=-1).astype(float)[None] for k, v in cols.items()}  # (1, units, days)

    # ---- truth for scoring (never shown to the readout)
    def true_effect(self, plan: dict) -> dict:
        """Expected (noiseless) extra D30 revenue and extra realised spend of the treated units over the treatment window, and their ratio."""
        w, nz = plan["windows"], self.cfg["noise"]
        inc = {u["unit_id"]: u["planned_incremental_spend"] for u in plan["units"] if u["arm"] == "treatment"}
        T = np.array([u in inc for u in self.unit_ids])
        extra = np.array([inc.get(u, 0.0) for u in self.unit_ids]) * nz["treated_delivery_mean"]
        rev = spend = 0.0
        for iso in np.datetime_as_string(np.arange(w["treatment_start"], P.add_days(w["treatment_end"], 1), dtype="datetime64[D]")):
            iso = str(iso)
            s1, s0 = self.b + extra, self.b
            r1 = self.expected_installs(s1) * self.expected_value(s1, iso)
            r0 = self.expected_installs(s0) * self.expected_value(s0, iso)
            rev += float((r1 - r0)[T].sum())
            spend += float(extra[T].sum())
        # reference: the same total extra spend spread proportionally over ALL units (what a channel-level step would be)
        total = extra[T].sum()
        pct = total / self.b.sum()
        s1 = self.b * (1 + pct)
        iso = plan["windows"]["treatment_start"]
        ch_rev = float(((self.expected_installs(s1) * self.expected_value(s1, iso)) - (self.expected_installs(self.b) * self.expected_value(self.b, iso))).sum())
        return {"true_iroas_treated_units": rev / spend, "true_extra_revenue_per_day": rev / w["treatment_days"], "true_extra_spend_per_day": spend / w["treatment_days"],
                "reference_channel_level_proportional_step_roas": ch_rev / total}

"""Generator validation: check that generated dev data satisfies the `calibration_intents` that were committed in
evals/scenarios/*.json BEFORE any data was generated. This is generator QA, not a decision: it may use hidden truth
and the full stored data. Decision-facing code must never import this module.

CAUTION: `true_step20_marginal_roas_over_target` is an APPROXIMATE CALIBRATION PROXY. It assumes value per install is
proportional to player quality, which is not exactly how the generator works. It is not simulation truth and must not be
used to score future experiments; Phase 4 uses the experiment simulator's own truth. Passing all intents means the
generator was tuned to meet them on the dev seed (disclosed amendments exist); it is not evidence about any model.
"""
import json
import operator
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.calendar import date_to_day, day_to_date
from src.data.metrics import store_fee

ROOT = Path(__file__).resolve().parents[2]
OPS = {">=": operator.ge, "<=": operator.le}
PAID = ["meta", "google_uac", "applovin", "unity", "tiktok"]


def load_scenarios(folder=None) -> dict:
    folder = Path(folder or ROOT / "evals" / "scenarios")
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}


def _load(conn):
    start = conn.execute("SELECT value FROM meta WHERE key='start_date'").fetchone()[0]
    players = pd.read_sql_query("SELECT player_id, channel, install_date, reported_at FROM players", conn)
    days = pd.read_sql_query("SELECT * FROM player_days", conn)
    spend = pd.read_sql_query("SELECT channel, spend_date, spend FROM spend_daily", conn)
    for df, col in ((players, "install_date"), (players, "reported_at"), (days, "event_date"), (days, "reported_at"), (spend, "spend_date")):
        df[col + "_day"] = pd.to_datetime(df[col]).sub(pd.Timestamp(start)).dt.days
    days = days.merge(players[["player_id", "channel", "install_date_day"]], on="player_id", how="left", validate="many_to_one")
    return start, players, days, spend


def _response_value(params, s):
    if params["type"] == "power":
        return params["a"] * s ** params["b"]
    return params["imax"] * s ** params["h"] / (params["k"] ** params["h"] + s ** params["h"])


def compute_intent_metrics(conn: sqlite3.Connection, hidden: dict) -> dict:
    fee = store_fee(conn)
    start, players, days, spend = _load(conn)
    days["net"] = days["iap_gross"] * (1 - fee) + days["ad_revenue"]

    def cohort_sum(n):
        d = days[days["day_n"] <= n]
        return d.groupby(["channel", "install_date_day"])["net"].sum()

    d3, d30, d180 = cohort_sum(3), cohort_sum(30), cohort_sum(180)
    installs = players.groupby(["channel", "install_date_day"]).size()
    spend_by = spend.set_index(["channel", "spend_date_day"])["spend"]
    spend_by.index.names = ["channel", "install_date_day"]

    hist = range(0, 185)  # cohorts whose D180 lies inside storage
    out = {}
    per = {}
    for ch in PAID:
        idx = pd.MultiIndex.from_product([[ch], hist], names=["channel", "install_date_day"])
        s30, s180, s3 = d30.reindex(idx).fillna(0).sum(), d180.reindex(idx).fillna(0).sum(), d3.reindex(idx).fillna(0).sum()
        n_inst = installs.reindex(idx).fillna(0).sum()
        sp = spend_by.reindex(idx).sum()
        target = s30 / s180
        per[ch] = {"avg_roas": s30 / sp, "target": target, "d3_per_install": s3 / n_inst, "mult": s30 / s3, "v_ref": s30 / n_inst, "installs": n_inst}

    for ch in PAID:  # per-channel summary for the calibration printout (not an intent)
        out[("_summary", ch)] = {k: float(v) for k, v in per[ch].items()}

    # --- avg ROAS over target, and the TRUE finite-step marginal ROAS over target (approximation: value per install ~ quality)
    for sid, ch in (("steady_winner", "meta"), ("front_loaded", "tiktok"), ("saturating_imperfect_curve", "applovin")):
        out[(sid, "avg_d30_roas_over_target")] = per[ch]["avg_roas"] / per[ch]["target"]
    for sid, ch in (("steady_winner", "meta"), ("saturating_imperfect_curve", "applovin")):
        t = hidden["channels"][ch]
        gamma, s0 = t["quality_gamma"], t["base_spend"]
        s1 = 1.2 * s0
        spends = np.array(t["daily_spend"][:185])
        w = np.array(t["daily_mean_installs_noiseless"][:185])
        q_ref = float((w * (spends / s0) ** (-gamma)).sum() / w.sum())
        value = lambda s: _response_value(t["response"], s) * per[ch]["v_ref"] * (s / s0) ** (-gamma) / q_ref
        out[(sid, "true_step20_marginal_roas_over_target")] = ((value(s1) - value(s0)) / (s1 - s0)) / per[ch]["target"]

    # --- front loaded
    others = [per[c]["mult"] for c in PAID if c != "tiktok"]
    out[("front_loaded", "d30_to_d3_multiplier_over_peer_median")] = per["tiktok"]["mult"] / float(np.median(others))
    ranking = sorted(PAID, key=lambda c: -per[c]["d3_per_install"])
    out[("front_loaded", "d3_net_revenue_per_install_rank")] = ranking.index("tiktok") + 1

    # --- tracking outage
    pl = hidden["planted"]
    o = pl["tracking_outage"]
    ch = o["channel"]
    ch_days = days[days["channel"] == ch]
    ch_inst = installs[ch].reindex(range(0, 365)).fillna(0).to_numpy()

    def activity_per_player(lo, hi):
        vals = []
        for e in range(lo, hi + 1):
            rows = ((ch_days["event_date_day"] == e)).sum()
            base = ch_inst[max(e - 30, 0) : e + 1].sum()
            vals.append(rows / base)
        return float(np.mean(vals))

    w0, w1 = o["start_day"], o["end_day"]
    before = activity_per_player(w0 - 14, w0 - 1)
    out[("tracking_outage", "outage_event_drop_fraction")] = 1 - activity_per_player(w0, w1) / before
    out[("tracking_outage", "outage_install_change_abs")] = abs(ch_inst[w0 : w1 + 1].mean() / ch_inst[w0 - 14 : w0].mean() - 1)

    # --- CPI spike
    c = pl["cpi_spike"]
    ch = c["channel"]
    sp_ch = spend[spend["channel"] == ch].set_index("spend_date_day")["spend"]
    inst_ch = installs[ch].reindex(range(0, 365)).fillna(0)
    cpi = sp_ch / inst_ch.replace(0, np.nan)
    out[("cpi_spike", "cpi_spike_ratio_to_trailing_median")] = float(cpi.loc[c["start_day"] : c["end_day"]].mean() / cpi.loc[c["start_day"] - 28 : c["start_day"] - 1].median())

    # --- delayed reporting
    days["lag"] = days["reported_at_day"] - days["event_date_day"]
    unity_lag = days.loc[days["channel"] == "unity", "lag"]
    out[("delayed_reporting", "min_report_lag_days")] = float(unity_lag.min())
    out[("delayed_reporting", "other_channels_max_report_lag_days")] = float(days.loc[days["channel"] != "unity", "lag"].max())
    # amended window (see delayed_reporting.json "amendment"): cohorts installed 4..8 days before as_of 301
    u = days[(days["channel"] == "unity") & (days["day_n"] <= 3) & (days["install_date_day"].between(293, 297))]
    u = u.merge(players[["player_id", "reported_at_day"]].rename(columns={"reported_at_day": "player_reported_day"}), on="player_id")
    visible = lambda asof: u[(u["reported_at_day"] <= asof) & (u["player_reported_day"] <= asof)]["net"].sum()
    out[("delayed_reporting", "naive_visible_d3_revenue_fraction")] = float(visible(301) / visible(336))

    # --- tiny sample + whale
    t = pl["tiny_sample_whale"]
    tk = installs["tiktok"].reindex(range(0, 365)).fillna(0)
    out[("tiny_sample_whale", "window_installs_per_day_max")] = float(tk.loc[t["start_day"] : t["end_day"]].max())
    tk_d3 = d3["tiktok"].reindex(range(0, 335)).fillna(0)
    normal = tk_d3.drop(index=range(t["start_day"], t["end_day"] + 1))
    out[("tiny_sample_whale", "whale_cohort_d3_net_over_normal_median")] = float(tk_d3.loc[t["whale_install_day"]] / normal.median())

    # --- regime shift (all paid channels pooled)
    paid_days = days[days["channel"].isin(PAID)]
    shift = pl["regime_shift"]["start_day"]
    pre = paid_days[paid_days["event_date_day"].between(shift - 56, shift - 1)]
    post = paid_days[paid_days["event_date_day"].between(shift, shift + 56)]
    out[("regime_shift", "iap_per_active_day_post_over_pre")] = float((post["iap_gross"].sum() / len(post)) / (pre["iap_gross"].sum() / len(pre)))
    out[("regime_shift", "ad_ecpm_post_over_pre")] = float((post["ad_revenue"].sum() / post["ad_impressions"].sum()) / (pre["ad_revenue"].sum() / pre["ad_impressions"].sum()))
    return out


def evaluate_intents(metrics: dict, scenarios: dict) -> list:
    """Return rows (scenario, metric, value, op, threshold, passed) for every committed calibration intent."""
    rows = []
    for sid, sc in scenarios.items():
        for intent in sc["calibration_intents"]:
            key = (sid, intent["metric"])
            value = metrics[key]
            rows.append((sid, intent["metric"], value, intent["op"], intent["value"], bool(OPS[intent["op"]](value, intent["value"]))))
    return rows

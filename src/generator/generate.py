"""Synthetic data generator. ALL DATA IS SYNTHETIC.

Output is a `GeneratedData` object: four decision-facing tables plus a separate `hidden_truth` dict.
Hidden truth (true response curves, planted event parameters, noiseless install means) must never be
written into the decision database; see scripts/generate_data.py.

Table grain:
  players          one row per installed player (attributed install)
  player_days      one row per player per ACTIVE day (a player who is inactive on a day has no row)
  spend_daily      one row per paid channel per calendar day
  cohort_reporting one row per cohort (channel x install date) per milestone (D3/D30/D180): declared completeness date
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.calendar import dates_for_days

ROOT = Path(__file__).resolve().parents[2]
MILESTONES = (3, 30, 180)


def load_config(path=None) -> dict:
    with open(path or ROOT / "config" / "generator.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class GeneratedData:
    players: pd.DataFrame
    player_days: pd.DataFrame
    spend_daily: pd.DataFrame
    cohort_reporting: pd.DataFrame
    meta: dict
    hidden_truth: dict


# ---------------------------------------------------------------- response curves and spend


def response_function(ch: dict):
    """Return (function spend -> expected installs, parameter dict). Calibrated so f(base_spend) = base_installs."""
    base, n0, r = ch["base_spend"], ch["base_installs"], ch["response"]
    if r["type"] == "power":
        a = n0 / base ** r["b"]
        return (lambda s: a * np.power(s, r["b"])), {"type": "power", "a": a, "b": r["b"]}
    if r["type"] == "hill":
        k = r["k_over_base"] * base
        h = r["h"]
        imax = n0 * (k**h + base**h) / base**h
        return (lambda s: imax * np.power(s, h) / (k**h + np.power(s, h))), {"type": "hill", "imax": imax, "k": k, "h": h}
    raise ValueError(f"unknown response type {r['type']}")


def make_spend(rng, ch: dict, cfg: dict, horizon: int) -> np.ndarray:
    """Daily spend: base x exp(slowly moving weekly level + daily noise), clipped. The variation is what
    makes a spend -> installs relationship estimable (a deliberate simplification, see README limitations)."""
    sp = cfg["spend_process"]
    weeks = (horizon + 6) // 7
    level = np.zeros(weeks)
    for w in range(1, weeks):
        level[w] = sp["weekly_level_persistence"] * level[w - 1] + rng.normal(0, sp["weekly_level_sd"])
    multiple = np.exp(level[np.arange(horizon) // 7] + rng.normal(0, sp["daily_noise_sd"], horizon))
    multiple = np.clip(multiple, sp["min_multiple_of_base"], sp["max_multiple_of_base"])
    return ch["base_spend"] * multiple


# ---------------------------------------------------------------- player simulation


def _lognormal_mean_one(rng, sd, size):
    return rng.lognormal(-0.5 * sd**2, sd, size)


def simulate_channel(rng, name, ch, installs, spend, cfg, id_offset, is_organic=False):
    """Simulate every player of one channel and their active days. Returns (players dict, rows dict)."""
    beh, planted = cfg["behaviour"], cfg["planted"]
    horizon = cfg["horizon_days"]
    max_n = cfg["max_activity_day_n"]
    lag = ch["report_lag"]

    install_day = np.repeat(np.arange(horizon), installs)  # sorted ascending
    n_players = len(install_day)

    # latent player traits (these never leave this function)
    if is_organic or ch["quality_gamma"] == 0:
        quality = np.ones(n_players)
    else:
        quality = np.power(spend[install_day] / ch["base_spend"], -ch["quality_gamma"])
    engagement = rng.gamma(beh["engagement_gamma_shape"], 1 / beh["engagement_gamma_shape"], n_players) * quality
    is_payer = rng.random(n_players) < np.clip(ch["payer_rate"] * quality, 0, 1)
    spend_scale = _lognormal_mean_one(rng, beh["spend_scale_sd"], n_players)
    big = is_payer & (rng.random(n_players) < beh["big_spender_share_of_payers"])
    lo, hi = beh["big_spender_multiplier_range"]
    spend_scale = np.where(big, spend_scale * rng.uniform(lo, hi, n_players), spend_scale)

    # attribution report time for the install itself
    player_lag = lag["min_days"] + rng.integers(0, lag["jitter_days"] + 1, n_players)

    shift = planted["regime_shift"]
    outage = planted["tracking_outage"]
    whale = planted["tiny_sample_whale"]
    whale_idx = None
    if name == whale["channel"]:
        pos = np.searchsorted(install_day, whale["whale_install_day"], side="left")
        if pos < n_players and install_day[pos] == whale["whale_install_day"]:
            whale_idx = pos
            is_payer[pos] = True

    pur = ch["purchase"]
    chunks = []
    for n in range(0, max_n + 1):
        cut = np.searchsorted(install_day, horizon - 1 - n, side="right")  # players whose day n is inside storage
        if cut == 0:
            break
        event_day = install_day[:cut] + n
        if n == 0:
            active = np.ones(cut, dtype=bool)
        else:
            p_active = ch["retention"]["r1"] * n ** (-ch["retention"]["alpha"]) * engagement[:cut]
            active = rng.random(cut) < np.clip(p_active, 0, 1)
        if whale_idx is not None and n == whale["whale_event_day_n"]:
            active[whale_idx] = True
        ids = np.nonzero(active)[0]
        ev = event_day[ids]

        # tracking outage: events on these days are lost for this channel (installs are NOT lost)
        if name == outage["channel"]:
            in_window = (ev >= outage["start_day"]) & (ev <= outage["end_day"])
            lost = in_window & (rng.random(len(ids)) < outage["event_loss_probability"])
            ids, ev = ids[~lost], ev[~lost]
        if len(ids) == 0:
            continue

        eng = engagement[ids]
        lam = max(beh["sessions_per_active_day"] - 1, 0) * np.sqrt(eng) * (beh["day0_session_bonus"] if n == 0 else 1.0)
        sessions = 1 + rng.poisson(lam)
        levels = rng.poisson(beh["levels_per_session"] * sessions)
        impressions = rng.poisson(beh["ad_impressions_per_session"] * sessions)
        after_shift = ev >= shift["start_day"]
        ecpm = ch["ecpm"] * _lognormal_mean_one(rng, beh["ecpm_noise_sd"], len(ids)) * np.where(after_shift, shift["ecpm_multiplier"], 1.0)
        ad_revenue = impressions * ecpm / 1000.0

        p_buy = pur["p_per_active_day"] * (1 + pur["front_boost"] * math.exp(-n / pur["front_tau_days"]))
        p_buy = np.clip(p_buy * np.where(after_shift, shift["iap_purchase_probability_multiplier"], 1.0), 0, 1)
        buys = is_payer[ids] & (rng.random(len(ids)) < p_buy)
        amount = pur["price"] * spend_scale[ids] * _lognormal_mean_one(rng, beh["purchase_amount_noise_sd"], len(ids))
        iap_gross = np.where(buys, amount, 0.0)
        if whale_idx is not None and n == whale["whale_event_day_n"]:
            iap_gross = np.where(ids == whale_idx, iap_gross + whale["whale_iap_gross"], iap_gross)

        row_lag = lag["min_days"] + rng.integers(0, lag["jitter_days"] + 1, len(ids))
        chunks.append(
            {
                "player_local": ids,
                "event_day": ev,
                "day_n": np.full(len(ids), n),
                "sessions": sessions,
                "levels": levels,
                "ad_impressions": impressions,
                "iap_gross": np.round(iap_gross, 2),
                "ad_revenue": np.round(ad_revenue, 6),
                "reported_day": ev + row_lag,
            }
        )

    rows = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    rows["player_id"] = rows.pop("player_local") + id_offset
    players = {
        "player_id": np.arange(n_players) + id_offset,
        "channel": np.full(n_players, name),
        "install_day": install_day,
        "reported_day": install_day + player_lag,
    }
    return players, rows


# ---------------------------------------------------------------- orchestration


def generate(cfg: dict, seed: int = None, install_scale: float = 1.0) -> GeneratedData:
    """Build all tables for one seed. `install_scale` shrinks install counts (tests only; 1.0 for real data)."""
    seed = cfg["seed"] if seed is None else seed
    horizon = cfg["horizon_days"]
    planted = cfg["planted"]
    channel_names = list(cfg["channels"])
    streams = np.random.SeedSequence(seed).spawn(len(channel_names) + 1)  # one independent stream per channel + organic
    weekday = np.array(cfg["weekday_install_multiplier"])[np.arange(horizon) % 7]

    all_players, all_rows, spend_frames, truth_channels = [], [], [], {}
    id_offset = 0
    for stream, name in zip(streams[:-1], channel_names):
        rng = np.random.default_rng(stream)
        ch = cfg["channels"][name]
        f, params = response_function(ch)
        spend = make_spend(rng, ch, cfg, horizon)
        if name == planted["tiny_sample_whale"]["channel"]:
            w = planted["tiny_sample_whale"]
            spend[w["start_day"] : w["end_day"] + 1] *= w["spend_multiplier"]
        spike = np.ones(horizon)
        if name == planted["cpi_spike"]["channel"]:
            c = planted["cpi_spike"]
            spike[c["start_day"] : c["end_day"] + 1] = c["install_multiplier"]
        mean_installs = f(spend) * weekday * spike * install_scale
        noisy = mean_installs * _lognormal_mean_one(rng, cfg["install_noise_sd"], horizon)
        installs = rng.poisson(noisy)

        players, rows = simulate_channel(rng, name, ch, installs, spend, cfg, id_offset)
        id_offset += len(players["player_id"])
        all_players.append(players)
        all_rows.append(rows)
        spend_frames.append(
            pd.DataFrame(
                {
                    "channel": name,
                    "spend_date": dates_for_days(cfg["start_date"], np.arange(horizon)),
                    "spend": np.round(spend, 2),
                    "reported_at": dates_for_days(cfg["start_date"], np.arange(horizon) + cfg["spend_report_lag_days"]),
                }
            )
        )
        truth_channels[name] = {
            "response": params,
            "quality_gamma": ch["quality_gamma"],
            "base_spend": ch["base_spend"],
            "daily_spend": np.round(spend, 4).tolist(),
            "daily_mean_installs_noiseless": np.round(mean_installs, 4).tolist(),
        }

    org = cfg["organic"]
    rng = np.random.default_rng(streams[-1])
    org_mean = org["base_installs"] * weekday * (1 + org["weekly_growth_per_day"] * np.arange(horizon)) * install_scale
    org_installs = rng.poisson(org_mean * _lognormal_mean_one(rng, cfg["install_noise_sd"], horizon))
    org_ch = dict(org, quality_gamma=0.0)
    players, rows = simulate_channel(rng, "organic", org_ch, org_installs, None, cfg, id_offset, is_organic=True)
    all_players.append(players)
    all_rows.append(rows)

    start = cfg["start_date"]
    players_df = pd.concat([pd.DataFrame(p) for p in all_players], ignore_index=True)
    players_df["install_date"] = dates_for_days(start, players_df.pop("install_day").to_numpy())
    players_df["reported_at"] = dates_for_days(start, players_df.pop("reported_day").to_numpy())
    players_df = players_df[["player_id", "channel", "install_date", "reported_at"]]

    days_df = pd.concat([pd.DataFrame(r) for r in all_rows], ignore_index=True)
    days_df["event_date"] = dates_for_days(start, days_df.pop("event_day").to_numpy())
    days_df["reported_at"] = dates_for_days(start, days_df.pop("reported_day").to_numpy())
    days_df = days_df[
        ["player_id", "event_date", "day_n", "sessions", "levels", "ad_impressions", "iap_gross", "ad_revenue", "reported_at"]
    ]

    # declared completeness: a milestone is complete at install + n + the channel's maximum reporting lag
    reporting = []
    for name in channel_names + ["organic"]:
        lag = (cfg["channels"][name] if name != "organic" else org)["report_lag"]
        sla = lag["min_days"] + lag["jitter_days"]
        for n in MILESTONES:
            d = np.arange(horizon - n)  # only cohorts whose day n lies inside storage
            reporting.append(
                pd.DataFrame(
                    {
                        "channel": name,
                        "install_date": dates_for_days(start, d),
                        "through_day": n,
                        "complete_at": dates_for_days(start, d + n + sla),
                    }
                )
            )

    meta = {
        "schema_version": "1",
        "start_date": start,
        "horizon_days": str(horizon),
        "last_event_date": dates_for_days(start, np.array([horizon - 1]))[0],
        "max_activity_day_n": str(cfg["max_activity_day_n"]),
        "store_fee": str(cfg["store_fee"]),
        "synthetic": "ALL DATA IS SYNTHETIC",
    }
    hidden = {
        "seed": seed,
        "install_scale": install_scale,
        "channels": truth_channels,
        "planted": planted,
        "organic_daily_mean_installs_noiseless": np.round(org_mean, 4).tolist(),
        "resolved_config": cfg,
    }
    return GeneratedData(
        players=players_df,
        player_days=days_df,
        spend_daily=pd.concat(spend_frames, ignore_index=True),
        cohort_reporting=pd.concat(reporting, ignore_index=True),
        meta=meta,
        hidden_truth=hidden,
    )

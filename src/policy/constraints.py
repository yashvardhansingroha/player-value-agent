"""Hard budget constraints, shared by the policy (planning), approval, apply and rollback. Pure functions, no I/O.

Budgets are daily spend per channel. The total is a CEILING with an unallocated reserve (ceiling - sum of budgets).
`check_constraints` returns a list of violation dicts; an empty list means every constraint holds.
"""
import hashlib
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-9


def load_policy(path=None) -> dict:
    with open(path or ROOT / "config" / "policy.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_forecast_config() -> dict:
    with open(ROOT / "config" / "forecast.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def policy_hash(policy: dict, forecast_cfg: dict = None) -> str:
    """Hash of the FULL decision configuration: the policy file and the forecast/quality file it depends on.
    Any change to either changes the hash, which makes earlier approvals stale."""
    forecast_cfg = load_forecast_config() if forecast_cfg is None else forecast_cfg
    blob = json.dumps({"policy": policy, "forecast": forecast_cfg}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def reserve(budgets: dict, ceiling: float) -> float:
    return float(ceiling) - float(sum(budgets.values()))


def check_constraints(current: dict, proposed: dict, ceiling: float, cfg: dict, check_cap: bool = True) -> list:
    """Violations of the hard constraints for moving from `current` to `proposed` budgets (dicts channel -> budget).
    Checks: finite and non-negative; floor; per-channel change cap (vs current); max channel share of the ceiling;
    total <= ceiling (the gap is the reserve); same set of channels."""
    c = cfg["constraints"]
    v = []
    if set(current) != set(proposed):
        v.append({"constraint": "channel_set", "detail": f"channels differ: {sorted(set(current) ^ set(proposed))}"})
        return v
    if not math.isfinite(ceiling) or ceiling < 0:
        v.append({"constraint": "ceiling", "detail": f"invalid ceiling {ceiling}"})
        return v
    for ch, new in sorted(proposed.items()):
        old = current[ch]
        if not isinstance(new, (int, float)) or not math.isfinite(new):
            v.append({"constraint": "finite", "channel": ch, "detail": f"budget {new!r} is not a finite number"})
            continue
        if new < 0:
            v.append({"constraint": "non_negative", "channel": ch, "detail": f"budget {new} is negative"})
        if new < c["min_spend_floor"] - EPS:
            v.append({"constraint": "min_spend_floor", "channel": ch, "detail": f"budget {new} is below the floor {c['min_spend_floor']}"})
        if check_cap and old > 0 and abs(new - old) / old > c["per_channel_change_cap"] + EPS:
            v.append({"constraint": "per_channel_change_cap", "channel": ch,
                      "detail": f"change {100 * (new - old) / old:+.1f}% exceeds the +/-{100 * c['per_channel_change_cap']:.0f}% cap (from {old} to {new})"})
        if ceiling > 0 and new / ceiling > c["max_channel_share"] + EPS:
            v.append({"constraint": "max_channel_share", "channel": ch,
                      "detail": f"budget {new} is {100 * new / ceiling:.1f}% of the ceiling, above {100 * c['max_channel_share']:.0f}%"})
    total = float(sum(x for x in proposed.values() if isinstance(x, (int, float)) and math.isfinite(x)))
    if total > ceiling + EPS:
        v.append({"constraint": "ceiling", "detail": f"total {total:.4f} exceeds the ceiling {ceiling:.4f} (reserve would be negative)"})
    return v

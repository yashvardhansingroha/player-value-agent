"""Hard budget constraints and the policy-configuration hash."""
import copy

from src.policy import constraints as cons

CFG = cons.load_policy()
CUR = {"a": 100.0, "b": 100.0, "c": 100.0}  # total 300
CEIL = 360.0  # reserve 60; 40% of the ceiling = 144


def kinds(proposed, current=CUR, ceiling=CEIL, **kw):
    return sorted(v["constraint"] for v in cons.check_constraints(current, proposed, ceiling, CFG, **kw))


def test_a_valid_change_has_no_violations_and_reserve_is_ceiling_minus_total():
    assert kinds({"a": 110.0, "b": 100.0, "c": 90.0}) == []
    assert cons.reserve(CUR, CEIL) == 60.0
    assert cons.reserve({"a": 150.0, "b": 100.0, "c": 110.0}, CEIL) == 0.0  # the ceiling can be used up exactly


def test_each_hard_constraint_is_detected_separately():
    assert kinds({"a": 125.0, "b": 100.0, "c": 100.0}) == ["per_channel_change_cap"]  # +25% > 20%
    assert kinds({"a": 75.0, "b": 100.0, "c": 100.0}) == ["per_channel_change_cap"]  # -25%
    assert kinds({"a": 120.0, "b": 100.0, "c": 100.0}) == []  # exactly 20% is allowed
    assert kinds({"a": 100.0, "b": 100.0, "c": 5.0}, current={"a": 100.0, "b": 100.0, "c": 5.5}) == ["min_spend_floor"]  # below floor 10
    assert kinds({"a": 120.0, "b": 100.0, "c": 100.0}, ceiling=300.0) == ["ceiling"]  # total 320 > 300, share 40% is allowed
    assert "max_channel_share" in kinds({"a": 120.0, "b": 100.0, "c": 100.0}, ceiling=290.0)  # 120/290 = 41.4% > 40%
    assert "non_negative" in kinds({"a": -1.0, "b": 100.0, "c": 100.0})
    assert "finite" in kinds({"a": float("nan"), "b": 100.0, "c": 100.0})
    assert "channel_set" in kinds({"a": 100.0, "b": 100.0})


def test_cap_check_can_be_skipped_only_for_initialisation():
    assert "per_channel_change_cap" not in kinds({"a": 150.0, "b": 100.0, "c": 100.0}, check_cap=False)


def test_policy_hash_changes_with_any_policy_or_forecast_setting():
    fc = cons.load_forecast_config()
    base = cons.policy_hash(CFG, fc)
    assert base == cons.policy_hash(copy.deepcopy(CFG), copy.deepcopy(fc))  # stable
    changed = copy.deepcopy(CFG)
    changed["constraints"]["per_channel_change_cap"] = 0.25
    assert cons.policy_hash(changed, fc) != base
    fc2 = copy.deepcopy(fc)
    fc2["quality"]["spike_ratio"] = 4.0
    assert cons.policy_hash(CFG, fc2) != base  # the forecast/quality file is part of the decision configuration

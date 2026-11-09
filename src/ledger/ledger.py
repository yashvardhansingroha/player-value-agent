"""Sandbox budget ledger: versioned budgets, proposals, approval binding, transactional apply, rollback and an audit log.
ALL DATA IS SYNTHETIC; nothing here moves real money.

Rules (each has a test):
  * The total is a CEILING with an unallocated reserve: sum(budgets) <= ceiling; reserve = ceiling - sum.
  * An approval is bound to the exact proposal hash, the ledger version, the policy configuration hash and the frozen plan hashes.
  * Apply re-checks everything inside one BEGIN IMMEDIATE transaction: the binding (stale approvals are rejected), the hard
    constraints (an approval can never override them), the plan gate for increases, and duplicate application (UNIQUE proposal id).
  * Rollback is per change identifier: it reverts a change only if every channel it touched still carries that change's id
    (no later change has been layered on top), and only if restoring the old budgets passes the hard constraints. Otherwise it
    reports `rollback_blocked` with the reason.
  * CONFIGURATION: the ledger holds ONE immutable snapshot of the policy and forecast configuration (stored as text, never a
    reference to the caller's dictionaries). Its hash is recomputed from that snapshot whenever it is used. The only way to change
    it is `reload_config` (audited); that changes the hash and so makes every earlier approval stale. Optionally an
    `active_config_provider` (for example "read the YAML files") is compared with the snapshot at approval and apply, so a change to
    the live configuration that was not reloaded is rejected as stale.
  * REVERSIBILITY: a change is refused at approval and again at apply unless its exact inverse also satisfies the hard constraints
    (for example a -20% cut has a +25% inverse and is refused). Rollback never bypasses the caps; later changes can still make a
    rollback impossible at runtime (`rollback_blocked`).
  * Increases need a REAL frozen experiment plan. `frozen_plans` stores the full canonical plan under its recomputed SHA-256; a bare
    hash cannot be registered. At approval AND apply the ledger recomputes the hash from the stored content and checks the plan is
    structurally valid (including that the treatment-unit budgets add up to exactly the approved step), adequate by its own recorded
    power numbers, and bound to this proposal item (channel, baseline and proposed budget, decision date, target), this ledger state
    and this policy configuration. `mechanics_test` remains only for testing ledger mechanics and is recorded as MECHANICS_TEST_APPLY.
  * EXPERIMENT CONFIGURATION: besides policy and forecast settings, the ledger holds an immutable snapshot of the AUTHORITATIVE
    experiment configuration (config/experiment.yaml unless one is passed). A plan is accepted only if that snapshot hashes to the
    plan's `experiment_config_hash` AND the plan follows it: adequacy standards, estimator, alpha, guardrails, allowed durations,
    horizon, unit counts, the shape of the recorded planning numbers, and a design fingerprint tying those numbers to this design.
    The plan can therefore not choose weaker standards than its declared configuration. (The experiment configuration is NOT part of
    `policy_hash`, so policy approvals and reports are unchanged; an approval is bound to it through the plan hash, which contains
    `experiment_config_hash`.)
    TRUST BOUNDARY: the ledger does not re-run the planning simulation, so it cannot prove the recorded keep rates were really
    produced by the study for this design; re-running that deterministic study is the independent check.
"""
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.experiments.plan import (config_hash, load_experiment_config, micro, plan_hash, validate_plan_against_config, validate_plan_binding,
                                  validate_plan_structure)
from src.ledger.proposal import Proposal, canonical, proposal_body, proposal_hash
from src.policy.constraints import check_constraints, load_forecast_config, policy_hash

SCHEMA = """
CREATE TABLE ledger_versions (
    version INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('INIT', 'APPLY', 'ROLLBACK', 'MECHANICS_TEST_APPLY')),
    change_id TEXT, proposal_id TEXT, policy_hash TEXT NOT NULL, ceiling REAL NOT NULL, note TEXT
);
CREATE TABLE ledger_budgets (
    version INTEGER NOT NULL REFERENCES ledger_versions (version),
    channel TEXT NOT NULL,
    budget REAL NOT NULL,
    channel_version INTEGER NOT NULL,   -- counts changes to this channel
    change_id TEXT NOT NULL,            -- the change that set this value
    PRIMARY KEY (version, channel)
);
CREATE TABLE proposals (
    proposal_id TEXT PRIMARY KEY, proposal_hash TEXT NOT NULL, body_json TEXT NOT NULL, ledger_version INTEGER NOT NULL,
    policy_hash TEXT NOT NULL, as_of TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('proposed', 'approved', 'rejected', 'applied')), created_at TEXT NOT NULL
);
CREATE TABLE approvals (
    approval_id INTEGER PRIMARY KEY AUTOINCREMENT, proposal_id TEXT NOT NULL, approver TEXT NOT NULL,
    decision TEXT NOT NULL CHECK (decision IN ('approve', 'reject')), proposal_hash TEXT NOT NULL,
    ledger_version INTEGER NOT NULL, policy_hash TEXT NOT NULL, plan_hashes_json TEXT NOT NULL, decided_at TEXT NOT NULL
);
CREATE TABLE frozen_plans (plan_hash TEXT PRIMARY KEY, plan_json TEXT NOT NULL, registered_by TEXT NOT NULL, registered_at TEXT NOT NULL);
CREATE TABLE applied_changes (
    change_id TEXT PRIMARY KEY,
    change_type TEXT NOT NULL CHECK (change_type IN ('APPLY', 'ROLLBACK')),
    proposal_id TEXT UNIQUE,            -- UNIQUE: a proposal can be applied at most once (NULL for rollbacks)
    rolls_back TEXT, kind TEXT, from_version INTEGER NOT NULL, to_version INTEGER NOT NULL, items_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'rolled_back')), rolled_back_by TEXT, applied_at TEXT NOT NULL, actor TEXT
);
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, event TEXT NOT NULL, proposal_id TEXT, change_id TEXT,
    ledger_version INTEGER, policy_hash TEXT, details_json TEXT NOT NULL
);
"""
PLAN_HASH = re.compile(r"^[0-9a-f]{64}$")


class Reject(Exception):
    def __init__(self, code, detail=None):
        super().__init__(code)
        self.code, self.detail = code, detail or {}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path, policy_cfg: dict, forecast_cfg: dict = None, allow_mechanics_test: bool = False, active_config_provider=None,
                 experiment_cfg: dict = None):
        self.path = str(path)
        new = not Path(self.path).exists() or self.path == ":memory:"
        self.conn = sqlite3.connect(self.path, isolation_level=None)  # explicit transactions
        if new:
            self.conn.executescript(SCHEMA)
            for t in ("audit_log", "approvals", "ledger_versions", "ledger_budgets", "frozen_plans"):
                for op in ("UPDATE", "DELETE"):
                    self.conn.execute(f"CREATE TRIGGER {t}_no_{op.lower()} BEFORE {op} ON {t} BEGIN SELECT RAISE(ABORT, '{t} is append-only'); END;")
            for t in ("proposals", "applied_changes"):
                self.conn.execute(f"CREATE TRIGGER {t}_no_delete BEFORE DELETE ON {t} BEGIN SELECT RAISE(ABORT, '{t} rows cannot be deleted'); END;")
        self._snapshot = self._freeze(policy_cfg, forecast_cfg, experiment_cfg)
        self._provider = active_config_provider
        self.allow_mechanics_test = allow_mechanics_test

    # ------------------------------------------------------------------ configuration snapshot
    @staticmethod
    def _freeze(policy_cfg: dict, forecast_cfg: dict = None, experiment_cfg: dict = None) -> str:
        """Serialise the configurations to text: nothing is shared with the caller's mutable dictionaries afterwards."""
        fc = load_forecast_config() if forecast_cfg is None else forecast_cfg
        ec = load_experiment_config() if experiment_cfg is None else experiment_cfg
        return json.dumps({"policy": policy_cfg, "forecast": fc, "experiment": ec}, sort_keys=True, separators=(",", ":"))

    @property
    def experiment_cfg(self) -> dict:
        """A fresh copy of the authoritative experiment configuration snapshot."""
        return json.loads(self._snapshot)["experiment"]

    @property
    def experiment_config_hash(self) -> str:
        return config_hash(self.experiment_cfg)

    @property
    def cfg(self) -> dict:
        """A fresh copy of the policy configuration snapshot. Changing the returned dict changes nothing in the ledger."""
        return json.loads(self._snapshot)["policy"]

    @property
    def fcfg(self) -> dict:
        return json.loads(self._snapshot)["forecast"]

    @property
    def policy_hash(self) -> str:
        """Hash of the snapshot, recomputed on every use (never cached)."""
        snap = json.loads(self._snapshot)
        return policy_hash(snap["policy"], snap["forecast"])

    def reload_config(self, policy_cfg: dict, forecast_cfg: dict = None, now: str = None, experiment_cfg: dict = None) -> dict:
        """The ONLY way to change the ledger's configuration. Audited. The policy hash changes, so earlier approvals become stale.
        `experiment_cfg=None` keeps the experiment configuration currently held (it does not silently fall back to the file)."""
        now = now or _now()
        old, old_exp = self.policy_hash, self.experiment_config_hash
        self._snapshot = self._freeze(policy_cfg, forecast_cfg, self.experiment_cfg if experiment_cfg is None else experiment_cfg)
        has_versions = self.conn.execute("SELECT COUNT(*) FROM ledger_versions").fetchone()[0] > 0
        self._audit("config_reloaded", now, details={"old_policy_hash": old, "new_policy_hash": self.policy_hash, "old_experiment_config_hash": old_exp,
                                                     "new_experiment_config_hash": self.experiment_config_hash}, version=self.head() if has_versions else None)
        return {"old_policy_hash": old, "new_policy_hash": self.policy_hash, "old_experiment_config_hash": old_exp, "new_experiment_config_hash": self.experiment_config_hash}

    def _check_active_config(self):
        """If an active-configuration provider was supplied, its hash must equal the snapshot's: a live change that was not reloaded is stale."""
        if self._provider is None:
            return
        p, f = self._provider()
        active = policy_hash(p, f)
        if active != self.policy_hash:
            raise Reject("stale_policy_config", {"ledger_snapshot_hash": self.policy_hash, "active_configuration_hash": active,
                                                 "action": "the active configuration changed after this ledger loaded it: reload_config and regenerate the proposal"})

    def close(self):
        self.conn.close()

    # ------------------------------------------------------------------ reading
    def head(self) -> int:
        r = self.conn.execute("SELECT MAX(version) FROM ledger_versions").fetchone()[0]
        if r is None:
            raise RuntimeError("ledger is not initialised")
        return r

    def state(self, version: int = None) -> dict:
        v = self.head() if version is None else version
        meta = self.conn.execute("SELECT ceiling, kind, change_id FROM ledger_versions WHERE version = ?", (v,)).fetchone()
        rows = self.conn.execute("SELECT channel, budget, channel_version, change_id FROM ledger_budgets WHERE version = ? ORDER BY channel", (v,)).fetchall()
        budgets = {c: b for c, b, _, _ in rows}
        total = float(sum(budgets.values()))
        return {"version": v, "kind": meta[1], "ceiling": meta[0], "budgets": budgets, "total": total, "reserve": meta[0] - total,
                "channels": {c: {"budget": b, "channel_version": cv, "change_id": cid} for c, b, cv, cid in rows}}

    def audit(self, event=None) -> list:
        q = "SELECT id, at, event, proposal_id, change_id, ledger_version, policy_hash, details_json FROM audit_log"
        rows = self.conn.execute(q + (" WHERE event = ?" if event else "") + " ORDER BY id", (event,) if event else ()).fetchall()
        return [dict(zip(("id", "at", "event", "proposal_id", "change_id", "ledger_version", "policy_hash", "details"), r[:7] + (json.loads(r[7]),))) for r in rows]

    def _audit(self, event, now, proposal_id=None, change_id=None, details=None, version=None):
        self.conn.execute("INSERT INTO audit_log (at, event, proposal_id, change_id, ledger_version, policy_hash, details_json) VALUES (?,?,?,?,?,?,?)",
                          (now, event, proposal_id, change_id, version, self.policy_hash, canonical(details or {})))

    # ------------------------------------------------------------------ initialise
    def initialize(self, budgets: dict, ceiling: float = None, now: str = None) -> dict:
        now = now or _now()
        ceiling = float(ceiling) if ceiling is not None else float(sum(budgets.values())) * (1 + self.cfg["constraints"]["ceiling_reserve_fraction"])
        bad = check_constraints(budgets, budgets, ceiling, self.cfg, check_cap=False)
        if bad:
            raise ValueError(f"initial budgets violate the hard constraints: {bad}")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("INSERT INTO ledger_versions VALUES (1, ?, 'INIT', 'INIT', NULL, ?, ?, 'sandbox initialisation')", (now, self.policy_hash, ceiling))
            self.conn.executemany("INSERT INTO ledger_budgets VALUES (1, ?, ?, 1, 'INIT')", sorted(budgets.items()))
            self._audit("ledger_initialized", now, details={"budgets": budgets, "ceiling": ceiling}, version=1)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return self.state()

    def register_frozen_plan(self, plan: dict, registered_by: str, now: str = None) -> str:
        """Store the full canonical plan under its RECOMPUTED hash and return the hash. An arbitrary hash can no longer be registered:
        the ledger keeps the content and recomputes the hash itself. Structurally invalid plans are refused; inadequate ones are
        stored (they are facts) but can never qualify an increase."""
        now = now or _now()
        errors = validate_plan_structure(plan)
        if errors:
            raise ValueError(f"plan is not structurally valid: {errors}")
        if plan["experiment_config_hash"] != self.experiment_config_hash:
            raise ValueError(f"plan was frozen under experiment configuration {plan['experiment_config_hash']}, but this ledger holds "
                             f"{self.experiment_config_hash}; load the plan's configuration first (reload_config)")
        errors = validate_plan_against_config(plan, self.experiment_cfg)
        if errors:
            raise ValueError(f"plan does not follow the ledger's experiment configuration: {errors}")
        h = plan_hash(plan)
        self.conn.execute("INSERT INTO frozen_plans VALUES (?,?,?,?)", (h, canonical(plan), registered_by, now))
        self._audit("plan_registered", now, details={"plan_hash": h, "channel": plan["channel"], "adequate": plan["power_assessment"]["adequate"],
                                                     "registered_by": registered_by})
        return h

    def get_plan(self, h: str):
        r = self.conn.execute("SELECT plan_json FROM frozen_plans WHERE plan_hash = ?", (h,)).fetchone()
        return None if r is None else json.loads(r[0])

    def record_event(self, event: str, now: str = None, change_id: str = None, details: dict = None):
        """Audit an event that happened outside the ledger's own transitions (for example an experiment readout decision)."""
        self._audit(event, now or _now(), None, change_id, details)

    # ------------------------------------------------------------------ propose / approve
    def create_proposal(self, proposal: Proposal, now: str = None) -> dict:
        now = now or _now()
        if proposal.ledger_version != self.head():
            raise ValueError(f"proposal was built on ledger version {proposal.ledger_version}, head is {self.head()}")
        body = proposal_body(proposal)
        h = proposal_hash(body)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("INSERT INTO proposals VALUES (?,?,?,?,?,?,'proposed',?)",
                              (proposal.proposal_id, h, canonical(body), proposal.ledger_version, proposal.policy_hash, proposal.as_of, now))
            self._audit("proposal_created", now, proposal.proposal_id, details={"proposal_hash": h, "source": proposal.source,
                        "actions": {i.channel: i.action for i in proposal.items}}, version=proposal.ledger_version)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return {"proposal_id": proposal.proposal_id, "proposal_hash": h}

    def _load_proposal(self, proposal_id):
        r = self.conn.execute("SELECT proposal_hash, body_json, ledger_version, policy_hash, status FROM proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
        return None if r is None else {"hash": r[0], "body": json.loads(r[1]), "ledger_version": r[2], "policy_hash": r[3], "status": r[4]}

    def verified_plan(self, h: str, channel: str = None) -> dict:
        """The registered plan `h`, after re-verifying it NOW: stored content hashes to `h` (not altered), is structurally valid (including
        the exact budget aggregation), follows this ledger's authoritative experiment configuration, and is adequate by its own recorded
        numbers. Raises Reject with the specific code otherwise. Used at approval, application and readout execution."""
        stored = self.get_plan(h)
        if stored is None:
            raise Reject("frozen_plan_not_registered", {"channel": channel, "plan_hash": h})
        recomputed = plan_hash(stored)
        if recomputed != h:
            raise Reject("plan_hash_mismatch", {"channel": channel, "plan_hash": h, "recomputed": recomputed,
                                                "detail": "the stored plan content does not hash to the registered hash (altered)"})
        errors = validate_plan_structure(stored)
        if errors:
            raise Reject("plan_invalid", {"channel": channel, "plan_hash": h, "errors": errors})
        if stored["experiment_config_hash"] != self.experiment_config_hash:
            raise Reject("experiment_config_not_loaded", {"channel": channel, "plan_hash": h, "plan_experiment_config_hash": stored["experiment_config_hash"],
                                                         "ledger_experiment_config_hash": self.experiment_config_hash,
                                                         "action": "load the experiment configuration the plan was frozen under into the ledger with reload_config"})
        errors = validate_plan_against_config(stored, self.experiment_cfg)
        if errors:
            raise Reject("plan_config_violation", {"channel": channel, "plan_hash": h, "errors": errors})
        if not stored["power_assessment"]["adequate"]:
            raise Reject("plan_inadequate", {"channel": channel, "plan_hash": h, "reasons": stored["power_assessment"]["reasons"]})
        return stored

    def verify_experiment_change(self, change_id: str, h: str) -> list:
        """Is `change_id` the applied, still-active ledger change that this plan authorised? Returns error strings (empty = it is). It must be
        an APPLY (not a mechanics test or a rollback) of a proposal whose ONLY changed item is this plan's channel, carrying this plan hash as
        an experiment-backed increase from the plan's baseline to its proposed budget, and its applied summary must agree."""
        plan = self.get_plan(h)
        if plan is None:
            return [f"plan {h} is not registered in the ledger"]
        row = self.conn.execute("SELECT change_type, kind, status, proposal_id, items_json FROM applied_changes WHERE change_id = ?", (change_id,)).fetchone()
        if row is None:
            return [f"change {change_id!r} does not exist in the ledger"]
        ctype, kind, status, proposal_id, items_json = row
        err = []
        if ctype != "APPLY" or kind != "APPLY":
            err.append(f"change {change_id} is a {ctype}/{kind}, not an experiment-backed APPLY")
        if status != "active":
            err.append(f"change {change_id} is {status}, not active")
        bs, ch = plan["budget_step"], plan["channel"]
        summary = json.loads(items_json)
        if set(summary) != {ch}:
            err.append(f"change {change_id} touched {sorted(summary)}, not only {ch}")
        elif micro(summary[ch]["from_budget"]) != micro(bs["baseline_daily_budget"]) or micro(summary[ch]["to_budget"]) != micro(bs["proposed_daily_budget"]):
            err.append(f"change {change_id} moved {ch} {summary[ch]['from_budget']} -> {summary[ch]['to_budget']}, not the plan's step "
                       f"{bs['baseline_daily_budget']} -> {bs['proposed_daily_budget']}")
        p = self._load_proposal(proposal_id) if proposal_id else None
        if p is None:
            err.append(f"change {change_id} has no stored proposal")
        else:
            its = [i for i in p["body"]["items"] if i.get("channel") == ch]
            if not its or its[0].get("qualification") != "experiment_backed" or its[0].get("experiment_plan_hash") != h:
                err.append(f"the proposal behind change {change_id} does not carry this plan's hash as an experiment-backed increase for {ch}")
        return err

    def _check_plan(self, item: dict, h: str, state: dict, proposal_as_of: str):
        """An experiment-backed increase needs the REAL plan: stored content that hashes to `h`, structurally valid (including the exact
        budget aggregation), following the ledger's authoritative experiment configuration, adequate by its own recorded numbers, and
        bound to this item, this ledger state and this policy configuration."""
        stored = self.verified_plan(h, item["channel"])
        errors = validate_plan_binding(stored, item, state, self.policy_hash, proposal_as_of)
        if errors:
            raise Reject("plan_binding_mismatch", {"channel": item["channel"], "plan_hash": h, "errors": errors})

    def _structural_and_constraint_checks(self, body, state, cfg):
        """Shared by approve and apply: item structure, plan gate, the hard constraints against the CURRENT ledger, and
        IMMEDIATE REVERSIBILITY (the exact inverse change must also satisfy them). `cfg` is the one configuration snapshot of this operation."""
        items = body["items"]
        current = state["budgets"]
        proposed = dict(current)
        for it in items:
            ch = it.get("channel")
            if ch not in current:
                raise Reject("invalid_action_budget", {"channel": ch, "detail": "unknown channel"})
            if it.get("current_budget") != current[ch]:
                raise Reject("proposal_current_budget_mismatch", {"channel": ch, "proposal_says": it.get("current_budget"), "ledger_has": current[ch]})
            new, act = it.get("new_budget"), it.get("action")
            ok_num = isinstance(new, (int, float)) and not isinstance(new, bool)
            if not ok_num:
                raise Reject("invalid_action_budget", {"channel": ch, "detail": f"new budget {new!r} is not a number"})
            if (act == "increase_test" and not new > current[ch]) or (act == "decrease" and not new < current[ch]) \
                    or (act in ("hold", "insufficient_evidence") and new != current[ch]) or act not in ("increase_test", "hold", "decrease", "insufficient_evidence"):
                raise Reject("invalid_action_budget", {"channel": ch, "detail": f"action {act!r} does not match budget change {current[ch]} -> {new}"})
            proposed[ch] = new
        changed = [it for it in items if it["new_budget"] != current[it["channel"]]]
        if not changed:
            raise Reject("no_budget_change", {})
        mechanics = False
        for it in changed:
            if it["action"] != "increase_test":
                continue
            q, plan = it.get("qualification"), it.get("experiment_plan_hash")
            if q == "mechanics_test":
                if not self.allow_mechanics_test:
                    raise Reject("mechanics_test_not_enabled", {"channel": it["channel"]})
                mechanics = True
            elif q == "experiment_backed" and plan and PLAN_HASH.match(plan):
                self._check_plan(it, plan, state, body["as_of"])
            else:
                raise Reject("increase_requires_frozen_plan", {"channel": it["channel"], "qualification": q,
                                                              "detail": "an increase needs a registered frozen experiment plan"})
        violations = check_constraints(current, proposed, state["ceiling"], cfg)
        if violations:
            raise Reject("constraint_violation", {"violations": violations})
        inverse = check_constraints(proposed, current, state["ceiling"], cfg)  # proposed -> current: what an immediate rollback would do
        if inverse:
            raise Reject("not_reversible", {"detail": "the exact inverse of this change would violate the hard constraints, so it could not be rolled back",
                                            "inverse_violations": inverse})
        return changed, proposed, mechanics

    def approve(self, proposal_id: str, approver: str, now: str = None) -> dict:
        now = now or _now()
        cfg, ph = self.cfg, self.policy_hash  # ONE configuration snapshot for every check and for the hash recorded in this approval
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            p = self._load_proposal(proposal_id)
            if p is None:
                raise Reject("proposal_not_found")
            if p["status"] != "proposed":
                raise Reject("not_open_for_approval", {"status": p["status"]})
            if proposal_hash(p["body"]) != p["hash"]:
                raise Reject("proposal_hash_mismatch")
            head = self.head()
            if p["ledger_version"] != head:
                raise Reject("stale_ledger_version", {"proposal_built_on": p["ledger_version"], "head": head})
            if p["policy_hash"] != ph:
                raise Reject("stale_policy_config", {"proposal_built_under": p["policy_hash"], "ledger_snapshot": ph})
            self._check_active_config()
            self._structural_and_constraint_checks(p["body"], self.state(), cfg)  # approval cannot be recorded for an invalid proposal
            plans = sorted({i["experiment_plan_hash"] for i in p["body"]["items"] if i.get("experiment_plan_hash")})
            self.conn.execute("INSERT INTO approvals (proposal_id, approver, decision, proposal_hash, ledger_version, policy_hash, plan_hashes_json, decided_at) "
                              "VALUES (?,?,?,?,?,?,?,?)", (proposal_id, approver, "approve", p["hash"], head, ph, json.dumps(plans), now))
            self.conn.execute("UPDATE proposals SET status = 'approved' WHERE proposal_id = ?", (proposal_id,))
            self._audit("approval_recorded", now, proposal_id, details={"approver": approver, "proposal_hash": p["hash"], "plan_hashes": plans}, version=head)
            self.conn.execute("COMMIT")
            return {"ok": True, "proposal_hash": p["hash"], "ledger_version": head}
        except Reject as r:
            self.conn.execute("ROLLBACK")
            self._audit("approval_refused", now, proposal_id, details={"code": r.code, **r.detail, "approver": approver})
            return {"ok": False, "code": r.code, "detail": r.detail}
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def reject(self, proposal_id: str, approver: str, reason: str, now: str = None) -> dict:
        now = now or _now()
        p = self._load_proposal(proposal_id)
        if p is None or p["status"] not in ("proposed", "approved"):
            return {"ok": False, "code": "not_open_for_rejection"}
        self.conn.execute("BEGIN IMMEDIATE")
        self.conn.execute("INSERT INTO approvals (proposal_id, approver, decision, proposal_hash, ledger_version, policy_hash, plan_hashes_json, decided_at) "
                          "VALUES (?,?,?,?,?,?,?,?)", (proposal_id, approver, "reject", p["hash"], self.head(), self.policy_hash, "[]", now))
        self.conn.execute("UPDATE proposals SET status = 'rejected' WHERE proposal_id = ?", (proposal_id,))
        self._audit("proposal_rejected", now, proposal_id, details={"approver": approver, "reason": reason}, version=self.head())
        self.conn.execute("COMMIT")
        return {"ok": True}

    # ------------------------------------------------------------------ apply
    def apply(self, proposal_id: str, now: str = None) -> dict:
        """Apply an approved proposal in ONE transaction. Everything is re-checked here, whatever approval says."""
        now = now or _now()
        cfg, ph = self.cfg, self.policy_hash  # ONE configuration snapshot for the hash comparison, the constraint checks and the version row
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            p = self._load_proposal(proposal_id)
            if p is None:
                raise Reject("proposal_not_found")
            if p["status"] == "applied":
                raise Reject("already_applied")
            appr = self.conn.execute("SELECT decision, proposal_hash, ledger_version, policy_hash, plan_hashes_json, approver FROM approvals "
                                     "WHERE proposal_id = ? ORDER BY approval_id DESC LIMIT 1", (proposal_id,)).fetchone()
            if p["status"] != "approved" or appr is None or appr[0] != "approve":
                raise Reject("not_approved", {"status": p["status"]})
            recomputed = proposal_hash(p["body"])
            if recomputed != p["hash"] or appr[1] != recomputed:
                raise Reject("proposal_hash_mismatch", {"stored": p["hash"], "recomputed": recomputed, "approved": appr[1]})
            head = self.head()
            if appr[2] != head:
                raise Reject("stale_ledger_version", {"approved_on": appr[2], "head": head})
            if appr[3] != ph or p["policy_hash"] != ph:
                raise Reject("stale_policy_config", {"approved_under": appr[3], "proposal_built_under": p["policy_hash"], "current": ph})
            self._check_active_config()
            plans = sorted({i["experiment_plan_hash"] for i in p["body"]["items"] if i.get("experiment_plan_hash")})
            if plans != json.loads(appr[4]):
                raise Reject("plan_hash_mismatch", {"approved": json.loads(appr[4]), "proposal": plans})
            state = self.state()
            changed, proposed, mechanics = self._structural_and_constraint_checks(p["body"], state, cfg)

            new_v = head + 1
            change_id = f"C-{new_v:04d}-{proposal_id[-10:]}"
            kind = "MECHANICS_TEST_APPLY" if mechanics else "APPLY"
            self.conn.execute("INSERT INTO ledger_versions VALUES (?,?,?,?,?,?,?,?)",
                              (new_v, now, kind, change_id, proposal_id, ph, state["ceiling"],
                               "MECHANICS TEST: not an experiment-backed increase" if mechanics else None))
            touched = {it["channel"]: it for it in changed}
            rows, summary = [], {}
            for ch, info in sorted(state["channels"].items()):
                if ch in touched:
                    rows.append((new_v, ch, proposed[ch], info["channel_version"] + 1, change_id))
                    summary[ch] = {"from_budget": info["budget"], "to_budget": proposed[ch], "from_channel_version": info["channel_version"],
                                   "to_channel_version": info["channel_version"] + 1, "from_change_id": info["change_id"]}
                else:
                    rows.append((new_v, ch, info["budget"], info["channel_version"], info["change_id"]))
            self.conn.executemany("INSERT INTO ledger_budgets VALUES (?,?,?,?,?)", rows)
            self.conn.execute("INSERT INTO applied_changes VALUES (?,?,?,?,?,?,?,?,'active',NULL,?,?)",
                              (change_id, "APPLY", proposal_id, None, kind, head, new_v, canonical(summary), now, appr[5]))
            self.conn.execute("UPDATE proposals SET status = 'applied' WHERE proposal_id = ?", (proposal_id,))
            after = self.state(new_v)
            self._audit("apply_succeeded", now, proposal_id, change_id, {"kind": kind, "approver": appr[5], "changes": summary,
                        "total_before": state["total"], "total_after": after["total"], "reserve_before": state["reserve"], "reserve_after": after["reserve"]}, new_v)
            self.conn.execute("COMMIT")
            return {"ok": True, "status": "applied", "change_id": change_id, "version": new_v, "kind": kind, "before": state, "after": after}
        except Reject as r:
            self.conn.execute("ROLLBACK")
            self._audit("apply_rejected", now, proposal_id, details={"code": r.code, **r.detail})
            return {"ok": False, "status": "rejected", "code": r.code, "detail": r.detail}
        except Exception as e:
            self.conn.execute("ROLLBACK")
            self._audit("apply_failed", now, proposal_id, details={"error": f"{type(e).__name__}: {e}"})
            raise

    # ------------------------------------------------------------------ rollback
    def rollback(self, change_id: str, requested_by: str, reason: str, now: str = None) -> dict:
        """Revert one applied change. Blocked (status `rollback_blocked`) if a later change touched any of its channels, or if
        restoring the old budgets would violate the hard constraints given the CURRENT ledger. All-or-nothing."""
        now = now or _now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            r = self.conn.execute("SELECT change_type, status, items_json FROM applied_changes WHERE change_id = ?", (change_id,)).fetchone()
            if r is None:
                raise Reject("change_not_found")
            if r[0] == "ROLLBACK":
                raise Reject("cannot_roll_back_a_rollback")
            if r[1] == "rolled_back":
                raise Reject("already_rolled_back")
            items = json.loads(r[2])
            state = self.state()
            conflicts = {}
            for ch, it in items.items():
                cur = state["channels"][ch]
                # `change_id` on a channel row names the change whose VALUE the channel currently holds. A later change replaces it;
                # rolling that later change back restores the earlier id, so changes can be unwound in reverse order.
                if cur["change_id"] != change_id:
                    conflicts[ch] = {"current_change_id": cur["change_id"], "current_channel_version": cur["channel_version"],
                                     "current_budget": cur["budget"], "expected_change_id": change_id}
            if conflicts:
                raise Reject("channel_changed_since", {"conflicts": conflicts})
            proposed = dict(state["budgets"])
            for ch, it in items.items():
                proposed[ch] = it["from_budget"]
            violations = check_constraints(state["budgets"], proposed, state["ceiling"], self.cfg)  # same caps as any change: never bypassed
            if violations:
                raise Reject("constraint_violation", {"violations": violations})

            new_v = state["version"] + 1
            rb_id = f"R-{new_v:04d}"
            self.conn.execute("INSERT INTO ledger_versions VALUES (?,?,'ROLLBACK',?,NULL,?,?,?)", (new_v, now, rb_id, self.policy_hash, state["ceiling"], f"rollback of {change_id}: {reason}"))
            rows, summary = [], {}
            for ch, info in sorted(state["channels"].items()):
                if ch in items:
                    # the restored value was set by the change that preceded this one, so that id goes back on the channel row;
                    # the rollback itself is recorded in ledger_versions / applied_changes / audit_log under rb_id
                    rows.append((new_v, ch, proposed[ch], info["channel_version"] + 1, items[ch]["from_change_id"]))
                    summary[ch] = {"from_budget": info["budget"], "to_budget": proposed[ch], "from_channel_version": info["channel_version"],
                                   "to_channel_version": info["channel_version"] + 1, "from_change_id": change_id}
                else:
                    rows.append((new_v, ch, info["budget"], info["channel_version"], info["change_id"]))
            self.conn.executemany("INSERT INTO ledger_budgets VALUES (?,?,?,?,?)", rows)
            self.conn.execute("INSERT INTO applied_changes VALUES (?,?,NULL,?,?,?,?,?,'active',NULL,?,?)",
                              (rb_id, "ROLLBACK", change_id, "ROLLBACK", state["version"], new_v, canonical(summary), now, requested_by))
            self.conn.execute("UPDATE applied_changes SET status = 'rolled_back', rolled_back_by = ? WHERE change_id = ?", (rb_id, change_id))
            after = self.state(new_v)
            self._audit("rollback_succeeded", now, None, rb_id, {"rolls_back": change_id, "requested_by": requested_by, "reason": reason, "changes": summary,
                        "total_before": state["total"], "total_after": after["total"], "reserve_before": state["reserve"], "reserve_after": after["reserve"]}, new_v)
            self.conn.execute("COMMIT")
            return {"ok": True, "status": "rolled_back", "change_id": rb_id, "version": new_v, "before": state, "after": after}
        except Reject as rj:
            self.conn.execute("ROLLBACK")
            blocked = rj.code in ("channel_changed_since", "constraint_violation")
            self._audit("rollback_blocked" if blocked else "rollback_rejected", now, None, change_id,
                        {"code": rj.code, "requested_by": requested_by, "reason": reason, **rj.detail})
            return {"ok": False, "status": "rollback_blocked" if blocked else "rejected", "code": rj.code, "detail": rj.detail}
        except Exception as e:
            self.conn.execute("ROLLBACK")
            self._audit("rollback_failed", now, None, change_id, {"error": f"{type(e).__name__}: {e}"})
            raise

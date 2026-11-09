"""Budget proposal model and its content hash. A proposal's hash covers its exact content (every item, budget, evidence and
the ledger version and policy hash it was built against), so an approval bound to that hash cannot be reused for different content.
ALL DATA IS SYNTHETIC; the ledger these go to is a SANDBOX."""
import hashlib
import json
import math
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

ACTIONS = ("increase_test", "hold", "decrease", "insufficient_evidence")
QUALIFICATIONS = ("none", "plan_required", "experiment_backed", "mechanics_test")


def jsonable(x):
    """Recursively convert numpy types and NaN/inf to JSON-safe values (NaN -> None)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class ProposalItem(BaseModel):
    channel: str
    action: Literal["increase_test", "hold", "decrease", "insufficient_evidence"]
    current_budget: float = Field(ge=0)
    new_budget: float = Field(ge=0)  # a negative budget cannot even be constructed here; the ledger re-checks raw stored data too
    # none: no experiment needed (hold, insufficient_evidence, decrease).
    # plan_required: an increase test whose pre-registered experiment plan does not exist yet (Phase 4): cannot be applied.
    # experiment_backed: carries a registered frozen plan hash. mechanics_test: ledger-mechanics testing only.
    qualification: Literal["none", "plan_required", "experiment_backed", "mechanics_test"] = "none"
    experiment_plan_hash: Optional[str] = None
    reason: str
    evidence: dict = Field(default_factory=dict)

    @field_validator("current_budget", "new_budget")
    @classmethod
    def _finite(cls, v):
        if not math.isfinite(v):
            raise ValueError("budget must be finite")
        return v


class Proposal(BaseModel):
    proposal_id: str
    as_of: str
    ledger_version: int
    policy_hash: str
    source: Literal["rules_only", "agent"] = "rules_only"
    items: list
    synthetic_notice: str = "SYNTHETIC DATA. SANDBOX LEDGER: nothing here moves real money."

    @field_validator("items")
    @classmethod
    def _items(cls, v):
        return [i if isinstance(i, ProposalItem) else ProposalItem(**i) for i in v]


def canonical(obj) -> str:
    return json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"))


def proposal_body(p: Proposal) -> dict:
    return jsonable(p.model_dump())


def proposal_hash(body: dict) -> str:
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def make_proposal_id(as_of: str, items: list, ledger_version: int, policy_hash: str) -> str:
    """Deterministic id from the content, so the same decision on the same state gets the same id."""
    blob = canonical({"as_of": as_of, "items": [i.model_dump() if isinstance(i, ProposalItem) else i for i in items],
                      "ledger_version": ledger_version, "policy_hash": policy_hash})
    return f"P-{as_of}-{hashlib.sha256(blob.encode()).hexdigest()[:10]}"

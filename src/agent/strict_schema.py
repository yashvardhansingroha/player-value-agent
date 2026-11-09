"""Provider schema for STRUCTURED FINALIZATION, derived from the existing output contract (src/agent/validator.py: AgentOutput). ALL DATA IS SYNTHETIC.

The contract stays the pydantic models; this module turns their JSON schema into the strict subset a provider's constrained decoding accepts:
  * every object lists ALL its properties as required and has additionalProperties = false (a field with a default becomes required; an optional field becomes a nullable union),
  * $ref / $defs are inlined (no references),
  * only these keywords survive: type, properties, required, additionalProperties, items, enum, anyOf, description (titles, defaults and any other keyword are dropped),
  * numbers stay numbers (new_budget), the action / qualification enums are unchanged.
A schema that is satisfied says only that the TEXT HAS THE RIGHT SHAPE. It says nothing about whether a citation is true, a population label right, an action permitted or a number correct:
the unchanged validator (src/agent/validator.py) checks all of that afterwards.

`check_strict(schema)` lists every violation of the restrictions we know of, so a provider rejection is caught in tests rather than in a live call. The restrictions come from the owner's
reading of the provider's documentation (strict mode: all properties required, additionalProperties false, unions via anyOf, no structured output together with tools or streaming); the
numeric limits below are conservative ASSUMPTIONS, not verified figures.
"""
import copy
import json

from src.agent.validator import AgentOutput

SCHEMA_NAME = "budget_proposal"
ALLOWED_KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum", "anyOf", "description"}
MAX_DEPTH = 5  # assumption: at most 5 levels of nested containers (objects/arrays); this contract uses exactly 5 (root, items[], item, claims[], claim)
MAX_PROPERTIES = 100  # assumption
MAX_SCHEMA_CHARS = 8000  # assumption: keep the schema small, it is paid for as prompt tokens

# Plain-language guidance for the model, attached as `description` (not an example, not a constraint). Keys are "<object>.<field>".
DESCRIPTIONS = {
    "Claim.tool_call_id": "the call id exactly as listed in the evidence index (for example tc_001)",
    "Claim.field": "the field path exactly as listed in the evidence index for that call",
    "Claim.value": "the exact value listed in the evidence index for that field (a number may be rounded to the decimals you write)",
    "Claim.population": "the population label exactly as shown in square brackets in the evidence index for that field",
    "Claim.channel": "the channel of the call, or null for a review-wide field",
    "AgentItem.new_budget": "the proposed daily budget as a number; for hold and insufficient_evidence it is the current budget",
    "AgentItem.experiment_plan_hash": "the plan_hash returned by plan_experiment for an experiment_backed increase, otherwise null",
}


def _inline(node, defs):
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline(copy.deepcopy(defs[node["$ref"].split("/")[-1]]), defs)
        return {k: _inline(v, defs) for k, v in node.items() if k != "$defs"}
    if isinstance(node, list):
        return [_inline(v, defs) for v in node]
    return node


def _strict(node, name=None):
    if not isinstance(node, dict):
        return node
    out = {}
    if "anyOf" in node:
        out["anyOf"] = [_strict(v) for v in node["anyOf"]]
    elif "enum" in node:
        out["type"] = node.get("type", "string")
        out["enum"] = list(node["enum"])
    elif node.get("type") == "object":
        props = node.get("properties", {})
        out["type"] = "object"
        out["properties"] = {k: _strict(v, f"{node.get('title', name)}.{k}") for k, v in props.items()}
        out["required"] = list(props)  # EVERY property is required in strict mode
        out["additionalProperties"] = False
    elif node.get("type") == "array":
        out["type"] = "array"
        out["items"] = _strict(node["items"], name)
    elif "type" in node:
        out["type"] = node["type"]
    if name in DESCRIPTIONS:
        out["description"] = DESCRIPTIONS[name]
    return out


def to_strict(model_schema: dict) -> dict:
    """Convert a pydantic JSON schema (with $defs) to the strict subset."""
    flat = _inline(copy.deepcopy(model_schema), model_schema.get("$defs", {}))
    title = flat.pop("title", None)
    return _strict(flat, title)


def provider_schema() -> dict:
    """The schema sent as response_format.json_schema.schema, derived from AgentOutput."""
    return to_strict(AgentOutput.model_json_schema())


def response_format(schema: dict = None, name: str = SCHEMA_NAME) -> dict:
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema or provider_schema()}}


def check_strict(schema: dict) -> list:
    """Every violation of the strict-subset restrictions (empty list = conforms)."""
    problems = []

    def walk(node, path, depth):
        if not isinstance(node, dict):
            problems.append(f"{path}: not an object")
            return
        if node.get("type") in ("object", "array"):  # nesting counts CONTAINERS (objects and arrays), not scalar leaves or the branches of a union
            depth += 1
            if depth > MAX_DEPTH:
                problems.append(f"{path}: containers nested deeper than {MAX_DEPTH}")
        extra = set(node) - ALLOWED_KEYWORDS
        if extra:
            problems.append(f"{path}: unsupported keyword(s) {sorted(extra)}")
        if "$ref" in node:
            problems.append(f"{path}: $ref is not allowed")
        t = node.get("type")
        if t == "object":
            props = node.get("properties", {})
            if node.get("additionalProperties") is not False:
                problems.append(f"{path}: additionalProperties must be false")
            if sorted(node.get("required", [])) != sorted(props):
                problems.append(f"{path}: every property must be required (missing {sorted(set(props) - set(node.get('required', [])))})")
            if len(props) > MAX_PROPERTIES:
                problems.append(f"{path}: more than {MAX_PROPERTIES} properties")
            for k, v in props.items():
                walk(v, f"{path}.{k}", depth)
        elif t == "array":
            if "items" not in node:
                problems.append(f"{path}: array without items")
            else:
                walk(node["items"], f"{path}[]", depth)
        elif "anyOf" in node:
            if not node["anyOf"]:
                problems.append(f"{path}: empty anyOf")
            for i, v in enumerate(node["anyOf"]):
                walk(v, f"{path}|{i}", depth)
        elif t is None and "enum" not in node:
            problems.append(f"{path}: no type")
        if "enum" in node and not node["enum"]:
            problems.append(f"{path}: empty enum")

    walk(schema, "$", 0)
    if len(json.dumps(schema, separators=(",", ":"))) > MAX_SCHEMA_CHARS:
        problems.append("schema longer than the conservative size bound")
    return problems

"""The compact AUTHORITATIVE evidence index handed to structured finalization. ALL DATA IS SYNTHETIC.

Built only from the ToolRunner's evidence log - the very records the validator checks every citation against - so investigation, finalization and validation share one source of truth.
For each SUCCESSFUL tool call it lists the call id, tool, channel and every citable scalar field (number, text, true/false) with its dotted path, its exact value and the
population label the tool attached to that path. Failed or denied calls are listed as such and are not evidence. Nothing is rounded, summarised or interpreted: floats are shown exactly as
stored. Null, object and list fields are not citable (the validator refuses them) and are not listed; text longer than MAX_TEXT characters is counted but not listed.
"""
import json

MAX_TEXT = 160


def scalar_leaves(result, prefix=""):
    """(dotted path, value) for every citable scalar inside `result`, in document order. Booleans are checked before numbers (bool is an int in Python)."""
    if isinstance(result, dict):
        for k, v in result.items():
            yield from scalar_leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(result, list):
        for i, v in enumerate(result):
            yield from scalar_leaves(v, f"{prefix}.{i}" if prefix else str(i))
    elif isinstance(result, (bool, int, float, str)) and prefix:
        yield prefix, result


def entries(runner) -> list:
    """Every listed fact as a dict {id, tool, channel, path, value, population}: the machine-readable form of the index (used by the tests)."""
    out = []
    for cid in runner.order:
        rec = runner.records[cid]
        if "error" in rec:
            continue
        for path, value in scalar_leaves(rec["result"]):
            if isinstance(value, str) and len(value) > MAX_TEXT:
                continue
            out.append({"id": cid, "tool": rec["tool"], "channel": rec.get("channel"), "path": path, "value": value, "population": runner.population_of(rec, path)})
    return out


def _fmt(v) -> str:
    return json.dumps(v, ensure_ascii=False) if isinstance(v, (str, bool)) else repr(v)


def build(runner) -> str:
    """The index as text: one block per call, facts grouped under their population label in square brackets."""
    lines = ["EVIDENCE INDEX (authoritative: produced by the deterministic tools in this run). Cite ONLY these call ids, field paths, values and population labels. Format: path=value; the [label] is the population."]
    by_id = {}
    for e in entries(runner):
        by_id.setdefault(e["id"], []).append(e)
    for cid in runner.order:
        rec = runner.records[cid]
        head = f"{cid} {rec['tool']}" + (f" channel={rec['channel']}" if rec.get("channel") else "") + (f" new_budget={rec['args']['new_budget']}" if "new_budget" in rec.get("args", {}) else "")
        if "error" in rec:
            lines.append(f"{cid} {rec['tool']}({json.dumps(rec.get('args', {}), separators=(',', ':'))}) -> ERROR {rec['error']['code']} (a failed call is not evidence; do not cite it)")
            continue
        lines.append(head)
        groups = {}
        for e in by_id.get(cid, []):
            groups.setdefault(e["population"], []).append(f"{e['path']}={_fmt(e['value'])}")
        omitted = sum(1 for p, v in _long_text(rec))
        for pop, facts in groups.items():
            lines.append(f" [{pop}] " + "; ".join(facts))
        if omitted:
            lines.append(f" (+{omitted} long text field(s) not listed)")
    return "\n".join(lines)


def _long_text(rec):
    for path, value in scalar_leaves(rec["result"]):
        if isinstance(value, str) and len(value) > MAX_TEXT:
            yield path, value

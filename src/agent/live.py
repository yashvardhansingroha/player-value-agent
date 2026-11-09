"""Live-verification helpers: choose a model FROM THE ACCOUNT'S OWN LISTING and verify it with the validated connectivity round trip before it is used.
Nothing here reads the key (the SDK client is passed in) and no model id is hard-coded or guessed. ALL DATA IS SYNTHETIC."""
from src.agent.gemini_check import run_check
from src.agent.llm import candidate_models, classify_rate_limit, list_generate_models

RATE_LIMIT_MARKERS = ("429", "RESOURCE_EXHAUSTED")


def select_verified_model(client, requested: str = None, max_tries: int = 4) -> dict:
    """List the models the account can use, then verify candidates (the requested one, else the best-ranked from the listing) with the plain / tool /
    round-trip check. Returns {"listed": [...], "tried": [{"model", "passed", "steps"}], "verified_model": id or None}. A model is `verified_model` only if
    ALL THREE steps passed against the live API."""
    out = {"listed": [], "tried": [], "verified_model": None}
    try:
        out["listed"] = list_generate_models(client)
    except Exception as exc:  # the account cannot even list models
        out["list_error"] = f"{type(exc).__name__}"
        return out
    order = [requested] if requested else candidate_models(out["listed"])[:max_tries]
    for model in order:
        res = run_check(client, model)
        out["tried"].append({"model": model, "passed": res.ok, "steps": [(n, p, d[:900]) for n, p, d in res.steps]})  # keep the provider's full message: it says WHICH quota
        if res.ok:
            out["verified_model"] = model
            break
        failure = " ".join(d for _, p, d in res.steps if not p)
        if any(marker in failure for marker in RATE_LIMIT_MARKERS):
            # a quota problem says nothing about the model, and every further candidate would spend more quota: stop and report it as what it is
            info = classify_rate_limit(failure)
            if info and info.kind == "billing":  # project-wide: every other model on this key would fail the same way
                out["stopped"] = (f"BILLING (HTTP 402) while checking {model}: the project's prepaid credits are depleted. This is an account/billing state, not a rate limit and not a model problem; "
                                  "it applies to every model on this key's project, so the remaining candidates were not tried. Use a key from a project that is in the free tier, or fund the project.")
                break
            kind = "DAILY QUOTA EXHAUSTED" if info and info.kind == "daily" else "RATE LIMITED"
            when = f" (the provider says it resets in about {info.retry_after / 3600:.1f} h)" if info and info.kind == "daily" and info.retry_after else ""
            out["stopped"] = (f"{kind} (HTTP 429) while checking {model}{when}: the quota, not the model, is the problem. The remaining candidates were not tried so they do not use "
                              "more quota; wait for the quota to reset (or use a key with a higher quota) and re-run.")
            break
    return out

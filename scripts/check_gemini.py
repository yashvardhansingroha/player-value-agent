"""Gemini connectivity check. Run it once GEMINI_API_KEY is set in your environment.

    python scripts/check_gemini.py                      # lists available models
    python scripts/check_gemini.py --model <model-id>   # also runs the validated plain / tool / round-trip check

Without a key it prints SKIPPED (live connectivity UNVERIFIED) and exits 0, because deterministic
development does not need the key. The key is read only from the environment and is never printed.
Exit code 1 means a live check ran and failed. The checking logic lives in src/agent/gemini_check.py.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai  # noqa: E402

from src.agent.gemini_check import run_check  # noqa: E402

KEY_VAR = "GEMINI_API_KEY"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="model id to test (see the listing)")
    args = parser.parse_args()

    key = os.environ.get(KEY_VAR)
    if not key:
        print(f"SKIPPED: {KEY_VAR} is not set. Live Gemini connectivity is UNVERIFIED.")
        return 0

    client = genai.Client(api_key=key)

    try:
        print("Models that support generateContent:")
        for m in client.models.list():
            if "generateContent" in (m.supported_actions or []):
                print("  ", m.name)
    except Exception as exc:
        print(f"FAILED: could not list models ({type(exc).__name__}): {str(exc).replace(key, '<key>')}")
        return 1

    if not args.model:
        print("No --model given: pick one from the list and re-run with --model <id>.")
        return 0

    result = run_check(client, args.model)
    for name, passed, detail in result.steps:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {detail.replace(key, '<key>')}")
    if result.ok:
        print(f"LIVE round trip OK with model {args.model}.")
        return 0
    print("FAILED: live check did not pass; round trip NOT verified.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

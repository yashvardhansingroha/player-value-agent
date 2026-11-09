"""Phase 0 smoke tests. These prove ONLY that the environment and config files load.
They say nothing about data, forecasts, policy or the agent."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_core_libraries_import():
    import numpy, pandas, pydantic, scipy, sklearn, sqlite3  # noqa: F401


def test_config_files_parse():
    for name in ("generator", "policy", "agent"):
        with open(ROOT / "config" / f"{name}.yaml", encoding="utf-8") as f:
            assert isinstance(yaml.safe_load(f), dict)


def test_no_api_key_in_config():
    text = (ROOT / "config" / "agent.yaml").read_text(encoding="utf-8")
    assert "AIza" not in text  # Google API keys start with this prefix

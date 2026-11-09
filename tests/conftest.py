import sqlite3
from pathlib import Path

import pytest

from scripts.generate_data import build
from src.generator.generate import load_config


@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    """Full dev-seed dataset (seed 1) generated once per test session into a temporary folder."""
    out = tmp_path_factory.mktemp("generated")
    report = build(seed=1, out_root=out)
    return {
        "db": Path(report["db_path"]),
        "hidden": Path(report["hidden_truth_path"]),
        "report": report,
        "cfg": load_config(),
        "out": out,
    }


@pytest.fixture(scope="session")
def small_dataset(tmp_path_factory):
    """Dev seed 1 at 30% install volume: a faster dataset for tests that run many reviews. Same generator, same structure."""
    out = tmp_path_factory.mktemp("generated_small")
    report = build(seed=1, out_root=out, install_scale=0.3)
    return {"db": Path(report["db_path"]), "out": out}


@pytest.fixture()
def conn(dataset):
    c = sqlite3.connect(dataset["db"])
    yield c
    c.close()

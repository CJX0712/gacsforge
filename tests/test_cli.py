"""CLI 冒烟测试。作者: 晨星"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cli_demo_smoke(tmp_path, monkeypatch, capsys):
    from semiforge.cli import main

    out = tmp_path / "benchmark.json"
    rc = main(["demo", "--out", str(out), "--datasets", "moons", "--seeds", "0,1,2"])
    assert rc == 0
    assert out.exists()
    import json

    with open(out, encoding="utf-8") as f:
        data = json.load(f)
    assert data["rows"]
    assert "gates" in data


def test_cli_bad_dataset():
    from semiforge.cli import main
    from semiforge.core.errors import DataError

    with pytest.raises((DataError, ValueError)):
        main(["demo", "--out", "x.json", "--datasets", "no_such"])

"""Smoke test: the ``inkcheck`` console script named in pyproject.toml exists, loads and runs (no network)."""

import json
import os
import subprocess
import sys
import tomllib
from importlib.metadata import distribution, entry_points
from pathlib import Path

import pytest

import inkcheck
from inkcheck.cli import main

SRC = Path(inkcheck.__file__).resolve().parents[1]
PYPROJECT = SRC.parent / "pyproject.toml"


def test_pyproject_entry_point_resolves():
    target = tomllib.loads(PYPROJECT.read_text())["project"]["scripts"]["inkcheck"]
    mod, func = target.split(":")
    assert getattr(__import__(mod, fromlist=[func]), func) is main


def test_main_info(capsys):
    assert main(["info", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["version"] == inkcheck.__version__
    assert out["reference_volume_id"] == "20250521140437"


def test_python_m_inkcheck():
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    r = subprocess.run([sys.executable, "-m", "inkcheck", "info", "--json"], capture_output=True, text=True,
                       env=env, timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["module"] == str(SRC / "inkcheck")


def test_installed_console_script():
    """Runs the real inkcheck executable when this checkout is the one installed (pip install -e)."""
    eps = entry_points(group="console_scripts", name="inkcheck")
    exe = Path(sys.executable).with_name("inkcheck")
    if not eps or not exe.exists():
        pytest.skip("inkcheck is not installed in this environment")
    url = json.loads(distribution("inkcheck").read_text("direct_url.json") or "{}").get("url", "")
    if url.startswith("file://") and Path(url[7:]).resolve() != SRC.parent:
        pytest.skip(f"the installed inkcheck is another checkout ({url})")
    assert next(iter(eps)).value == "inkcheck.cli:main"
    r = subprocess.run([str(exe), "info", "--json"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["version"] == inkcheck.__version__

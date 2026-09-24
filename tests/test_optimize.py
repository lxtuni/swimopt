# -*- coding: utf-8 -*-
"""The search engine: parallel equals serial, outputs, and the panel's parsing."""
import copy
import json
import os

import pytest

import optimize


def _run(cfg, tmp_path, tag, workers, budget=12):
    c = copy.deepcopy(cfg)
    c.update(outdir=str(tmp_path / tag), workers=workers, popsize=4, seed=7)
    path = tmp_path / f"{tag}.json"
    path.write_text(json.dumps(c), encoding="utf-8")
    optimize.main(str(path), budget)
    out = tmp_path / tag
    return (json.loads((out / "best.json").read_text(encoding="utf-8")),
            (out / "log.csv").read_text(encoding="utf-8"),
            json.loads((out / "convergence.json").read_text(encoding="utf-8")),
            json.loads((out / "run_config.json").read_text(encoding="utf-8")))


def test_parallel_search_equals_serial_search(cfg, tmp_path):
    """Rollouts are deterministic and independent, so farming them out to worker
    processes must not change a single number."""
    best1, log1, conv1, _ = _run(cfg, tmp_path, "serial", workers=1)
    best2, log2, conv2, _ = _run(cfg, tmp_path, "parallel", workers=2)
    best1.pop("elapsed_s", None)
    best2.pop("elapsed_s", None)
    assert best1 == best2
    assert log1 == log2
    assert conv1 == conv2


def test_outputs_carry_provenance(cfg, tmp_path):
    best, log, conv, run_cfg = _run(cfg, tmp_path, "prov", workers=1)
    assert len(best["x"]) == best["dim"] == 35             # full vector, not a subset
    assert run_cfg["config"]["seed"] == 7 and "mujoco" in run_cfg
    assert len(conv["history"]) == len(conv["sigma"]) == 3
    header = log.splitlines()[0].split(",")
    assert header[:len(optimize.LOG_FIELDS)] == optimize.LOG_FIELDS
    assert "feasible" in best and "heading_rms" in best


def test_seed_zero_warns(capsys):
    optimize.cma_options({"seed": 0}, 10)
    assert "NOT be reproducible" in capsys.readouterr().out


def test_printed_lines_are_what_the_panel_parses(cfg, tmp_path, capsys):
    import ui
    _run(cfg, tmp_path, "parse", workers=1, budget=8)
    out = capsys.readouterr().out.splitlines()
    cands = [ui.CANDIDATE_RE.search(l) for l in out if l.lstrip().startswith("#")]
    gens = [l for l in out if l.startswith("--- gen")]
    assert len(cands) == 8 and all(cands)
    assert [int(m.group(1)) for m in cands] == list(range(1, 9))
    assert all(ui.GENERATION_RE.search(l) and ui.SIGMA_RE.search(l) for l in gens)


def test_worker_count(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 16)
    assert optimize.n_workers({"popsize": 10}) == 10
    assert optimize.n_workers({"popsize": 40}) == 16
    assert optimize.n_workers({"workers": 1}) == 1

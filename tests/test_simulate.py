# -*- coding: utf-8 -*-
"""Rollouts: determinism, the settle cache, the measurement window, provenance."""
import copy
import math

import mujoco
import numpy as np
import pytest

from conftest import make_swimmer, write_mjcf
from simulate import Swimmer, record_best, strip_jsonc
import json


def _same(a, b):
    keys = [k for k in a if k != "traj"]
    return all(a[k] == b[k] for k in keys)


def test_rollouts_are_deterministic(cfg):
    sw = make_swimmer(cfg)
    x = np.random.default_rng(2).uniform(0, 1, sw.gait.dim)
    assert _same(sw.rollout(x), sw.rollout(x))


def test_settle_cache_is_bit_identical(cfg):
    """The second rollout restores the settled state instead of simulating it."""
    x = np.random.default_rng(3).uniform(0, 1, 35)
    fresh = make_swimmer(cfg)
    first = fresh.rollout(x)                 # simulates the settle, fills the cache
    assert fresh._settled is not None
    cached = fresh.rollout(x)                # restores from the cache
    other = make_swimmer(cfg)
    other.rollout(np.full(35, 0.5))          # a different candidate fills its cache
    assert _same(first, cached)
    assert _same(first, other.rollout(x))


def test_window_is_whole_strokes_and_never_shorter(cfg):
    sw = make_swimmer(cfg)
    for x0 in (0.0, 0.3, 0.77, 1.0):
        x = sw.gait.x0()
        x[0] = x0
        p = sw.gait.decode(x)
        w = sw.window(p)
        strokes = w * p["freq"]
        assert w >= sw.T - 1e-9
        assert strokes == pytest.approx(round(strokes))
    sw.whole_strokes = False
    assert sw.window(p) == sw.T


def test_attitude_and_heading_extraction(cfg):
    """The hot loop reads roll/pitch/yaw from flat xmat indices; check them against
    the matrix form for arbitrary orientations."""
    sw = make_swimmer(cfg)
    m, d = sw.model, sw.data
    rng = np.random.default_rng(4)
    for _ in range(20):
        q = rng.normal(size=4)
        d.qpos[3:7] = q / np.linalg.norm(q)
        mujoco.mj_forward(m, d)
        R = d.xmat[sw.trunk_id].reshape(3, 3)
        assert sw._yaw() == pytest.approx(np.arctan2(R[1, 0], R[0, 0]))
        flat = d.xmat[sw.trunk_id]
        assert abs(math.asin(min(1, max(-1, -flat[6])))) == pytest.approx(
            abs(np.arcsin(np.clip(-R[2, 0], -1, 1))))
        assert abs(math.atan2(flat[7], flat[8])) == pytest.approx(
            abs(np.arctan2(R[2, 1], R[2, 2])))


def test_best_record_replays_the_same_gait_after_a_preset_change(cfg):
    """best.json used to hold only the optimized subset, so changing the phase preset
    after a run silently replayed a different gait."""
    c = copy.deepcopy(cfg)
    c["gait"]["optimize"] = {"freq": True, "duty": True, "amp": True,
                             "phase": False, "offset": False}
    c["gait"]["preset_phase"] = "diag"
    run = make_swimmer(c)
    x = run.gait.x0_opt() + 0.1
    rec = record_best(run.rollout(x), x, run.gait)
    assert len(rec["x"]) == run.gait.dim and rec["preset"] == "diag"

    c["gait"]["preset_phase"] = "wave"
    later = make_swimmer(c)
    assert np.allclose(later.gait.decode(rec["x"])["PH"][0],
                       run.gait.decode(x)["PH"][0])


def test_body_length_is_measured_in_the_world_frame(cfg):
    sw = make_swimmer(cfg)
    d = mujoco.MjData(sw.model)
    mujoco.mj_forward(sw.model, d)
    robot = [g for g in range(sw.model.ngeom) if sw.model.geom_bodyid[g] != 0]
    xs = [d.geom_xpos[g][0] + s * sw.model.geom_rbound[g] for g in robot for s in (-1, 1)]
    assert sw.body_len == pytest.approx(max(xs) - min(xs))


def test_a_welded_robot_is_reported(tmp_path, cfg):
    xml = write_mjcf(tmp_path, """<body name="trunk" pos="0 0 0">
        <geom name="trunk_g" type="box" size="0.1 0.05 0.02"/></body>""")
    c = copy.deepcopy(cfg)
    c["model"] = xml
    sw = Swimmer(xml, c)
    assert not sw.free_base
    assert "NO free joint" in sw.info()


def test_jsonc_keeps_slashes_inside_strings():
    text = '{"url": "http://a/b", // comment\n "n": 1 /* block */}'
    assert json.loads(strip_jsonc(text)) == {"url": "http://a/b", "n": 1}

# -*- coding: utf-8 -*-
"""Gait parameterization: vector layout, freezing, and the control law."""
import copy

import numpy as np
import pytest

from conftest import make_swimmer


def _frozen(cfg, preset="diag"):
    c = copy.deepcopy(cfg)
    c["gait"]["optimize"] = {"freq": True, "duty": True, "amp": True,
                             "phase": False, "offset": False}
    c["gait"]["preset_phase"] = preset
    return make_swimmer(c)


def test_dimensions_follow_the_groups(cfg):
    g = make_swimmer(cfg).gait
    assert (g.n, g.n_amp, g.n_off) == (12, 3, 3)
    assert g.dim == 2 + g.H * (g.n_amp + g.n_phase) + g.n_off == 35


def test_reduced_and_full_vectors_decode_identically(cfg):
    g = _frozen(cfg).gait
    assert g.dim_opt == 8
    xr = g.x0_opt() + 0.1
    a, b = g.decode(xr), g.decode(g.expand(xr))
    assert a["freq"] == b["freq"]
    for key in ("A", "PH"):
        for u, v in zip(a[key], b[key]):
            assert np.array_equal(u, v)
    assert np.array_equal(a["offs"], b["offs"])


def test_wrong_length_vector_is_rejected(cfg):
    g = _frozen(cfg).gait
    with pytest.raises(ValueError):
        g.expand(np.zeros(3))


def test_preset_only_moves_the_start_point_when_phase_is_frozen(cfg):
    """The panel always writes a preset; it must not leak into a search that
    optimizes phase, or the panel and the command line disagree."""
    c = copy.deepcopy(cfg)
    c["gait"]["preset_phase"] = "wave"
    assert np.array_equal(make_swimmer(c).gait.base_x, make_swimmer(cfg).gait.base_x)
    assert not np.array_equal(_frozen(cfg, "wave").gait.base_x,
                              make_swimmer(cfg).gait.base_x)


def test_vectorized_control_matches_the_scalar_formula(cfg):
    g = make_swimmer(cfg).gait
    rng = np.random.default_rng(1)
    p = g.decode(rng.uniform(0, 1, g.dim))
    for t in np.linspace(0, 4, 57):
        ramp = min(1.0, t / g.ramp_t)
        cyc = (p["freq"] * t) % 1.0
        d = p["duty"]
        th = cyc / d * np.pi if cyc < d else np.pi + (cyc - d) / (1 - d) * np.pi
        ref = np.array([p["offs"][g._off_slot[i]]
                        + sum(ramp * p["A"][h][g._amp_slot[i]]
                              * np.sin((h + 1) * th + p["PH"][h][i]) for h in range(g.H))
                        for i in range(g.n)])
        assert np.allclose(g.ctrl(p, t), ref, atol=1e-12)


def test_period_is_one_over_frequency(cfg):
    g = make_swimmer(cfg).gait
    p = g.decode(g.x0())
    assert g.period(p) == pytest.approx(1.0 / p["freq"])

# -*- coding: utf-8 -*-
"""Legs from geometry, gait families, and classifying what the search found."""
import copy

import mujoco
import numpy as np
import pytest

from conftest import make_swimmer, write_mjcf
from gait import FAMILIES, SineGait, family_key, limb_layout
from simulate import record_best, swimmer_for_result
import view


def _family(cfg, fam, **gait):
    c = copy.deepcopy(cfg)
    c["gait"]["family"] = fam
    c["gait"].update(gait)
    return make_swimmer(c)


def test_legs_and_joint_types_come_from_geometry(cfg):
    sw = make_swimmer(cfg)
    g = sw.gait
    for i, name in enumerate(g.names):          # toy_quad: Joint_<leg><joint>.1
        assert g.legs[i] == name[6:8]
        assert g.depth[i] == int(name[8]) - 1


def test_geometry_wins_over_uninformative_names(tmp_path):
    """Legs named A-D, nothing in the names says which is which."""
    legs = ""
    for nm, x, y in (("A", 0.1, 0.05), ("B", 0.1, -0.05), ("C", -0.1, 0.05), ("D", -0.1, -0.05)):
        legs += f"""<body name="leg{nm}" pos="{x} {y} 0"><joint name="j{nm}" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.05" size="0.005"/>
          <body name="foot{nm}" pos="0 0 -0.05"><joint name="k{nm}" axis="0 1 0"/>
            <geom type="box" size="0.004 0.02 0.02"/></body></body>"""
    acts = "".join(f'<position name="{j}{nm}" joint="{j}{nm}" kp="5"/>'
                   for nm in "ABCD" for j in "jk")
    path = write_mjcf(tmp_path, f"""<body name="trunk"><freejoint/>
        <geom type="box" size="0.12 0.06 0.02"/>{legs}</body>""")
    text = open(path, encoding="utf-8").read().replace("</mujoco>",
                                                       f"<actuator>{acts}</actuator></mujoco>")
    open(path, "w", encoding="utf-8").write(text)
    legs_found, depth = limb_layout(mujoco.MjModel.from_xml_path(path))
    assert legs_found == ["FL", "FL", "FR", "FR", "BL", "BL", "BR", "BR"]
    assert list(depth) == [0, 1] * 4


def test_family_mode_searches_joint_types_not_actuators(cfg):
    assert make_swimmer(cfg).gait.dim == 35
    g = _family(cfg, "trot").gait
    assert g.family == "diag" and g.dim == 2 + 2 * (3 + 3) + 3 == 17


def test_a_family_delays_whole_legs_in_time(cfg):
    """In a trot the front-right leg does exactly what the front-left did half a stroke
    earlier -- even with an asymmetric duty cycle, which is why the delay is applied
    to time, before the duty warp, rather than as a phase offset."""
    g = _family(cfg, "diag").gait
    rng = np.random.default_rng(5)
    p = g.decode(rng.uniform(0, 1, g.dim))
    T = g.period(p)
    fl = [i for i in range(g.n) if g.legs[i] == "FL"]
    fr = [i for i in range(g.n) if g.legs[i] == "FR"]
    for t in np.linspace(g.ramp_t + T, g.ramp_t + 3 * T, 17):
        assert np.allclose(g.ctrl(p, t)[fr], g.ctrl(p, t - 0.5 * T)[fl], atol=1e-9)


@pytest.mark.parametrize("fam", sorted(FAMILIES))
def test_each_family_is_recognised_as_itself(cfg, fam):
    g = _family(cfg, fam).gait
    rng = np.random.default_rng(6)
    st = g.structure(rng.uniform(0, 1, g.dim))
    assert st["nearest"] == fam and st["deviation"] < 1e-9


def test_a_hand_built_trot_is_recognised_as_a_trot(cfg):
    g = make_swimmer(cfg).gait
    st = g.structure(view.demo_params(g))
    assert st["nearest"] == "diag" and st["deviation"] < 1e-9


def test_free_mode_is_unchanged(cfg):
    """Free phases must decode exactly as before families existed."""
    g = make_swimmer(cfg).gait
    x = np.random.default_rng(7).uniform(0, 1, g.dim)
    p = g.decode(x)
    k = 2 + g.n_amp
    assert np.allclose(p["PH"][0], x[k:k + g.n] * 2 * np.pi)
    assert p["_u_shift"] is None


def test_bad_family_names_are_rejected(cfg):
    with pytest.raises(ValueError):
        family_key("gallop")
    assert family_key("free") is None and family_key("trot") == "diag"


def test_a_result_replays_under_its_own_family(cfg):
    """A parameter vector means a different gait under a different family, so a
    result must be replayed with the family it was found in, whatever the config
    says now."""
    run = _family(cfg, "diag")
    x = np.random.default_rng(8).uniform(0, 1, run.gait.dim)
    rec = record_best(run.rollout(x), x, run.gait)
    assert rec["family"] == "diag"
    c = copy.deepcopy(cfg)
    c["gait"]["family"] = "wave"                  # the user changed the dropdown since
    later = swimmer_for_result(c, rec)
    assert later.gait.family == "diag"
    assert np.allclose(later.gait.decode(rec["x"])["PH"][0], run.gait.decode(x)["PH"][0])


def test_results_from_before_families_still_replay(cfg):
    """Older best.json files carry "preset" and no "family". They were free-phase
    unless phase was frozen, and must decode that way whatever the panel says now."""
    free = make_swimmer(cfg)
    x = np.random.default_rng(9).uniform(0, 1, free.gait.dim)
    old = {"x": list(x), "preset": None, "optimized": ["freq", "duty", "amp", "phase",
                                                        "offset"]}
    c = copy.deepcopy(cfg)
    c["gait"]["family"] = "diag"
    sw = swimmer_for_result(c, old)
    assert sw.gait.family is None and sw.gait.dim == 35
    old_frozen = {"x": [0.5] * 17, "preset": "fb", "optimized": ["freq", "duty", "amp"]}
    assert swimmer_for_result(c, old_frozen).gait.family == "fb"


def test_legacy_preset_still_only_applies_with_phase_frozen(cfg):
    c = copy.deepcopy(cfg)
    c["gait"]["preset_phase"] = "wave"
    assert make_swimmer(c).gait.family is None
    c["gait"]["optimize"] = {"phase": False}
    assert make_swimmer(c).gait.family == "wave"

# -*- coding: utf-8 -*-
"""Hydrodynamics: the fast path, the lift model, and conservation checks."""
import copy

import mujoco
import numpy as np
import pytest

from conftest import make_swimmer, write_mjcf
from hydro import HydroModel


def _random_state(m, d, rng):
    mujoco.mj_resetData(m, d)
    d.qpos[2] += rng.uniform(-0.03, 0.03)            # bob across the waterline
    q = rng.normal(size=4)
    d.qpos[3:7] = q / np.linalg.norm(q)
    d.qpos[7:] = rng.uniform(-1.0, 1.0, size=m.nq - 7)
    d.qvel[:] = rng.normal(scale=0.5, size=m.nv)
    mujoco.mj_forward(m, d)


@pytest.mark.parametrize("lift", [False, True])
def test_fast_path_matches_scalar_reference(cfg, lift):
    """apply() is an optimized rewrite; _apply_slow is the readable statement."""
    sw = make_swimmer(cfg, hydro={"lift": lift})
    m, d = sw.model, sw.data
    rng = np.random.default_rng(0)
    for _ in range(100):
        _random_state(m, d, rng)
        sw.hydro.apply(d)
        fast = d.xfrc_applied.copy()
        sw.hydro._apply_slow(d)
        slow = d.xfrc_applied.copy()
        assert np.abs(fast - slow).max() <= 1e-12 * max(np.abs(slow).max(), 1.0)


# ---------- lift model on a single analytic plate ----------
def _plate(cl=1.1, area=0.01, rho=1000.0):
    h = HydroModel.__new__(HydroModel)             # no model needed for _lift
    h.NRM = np.array([[1.0, 0.0, 0.0]])            # plate normal = local x
    h.K_lift = np.array([rho * cl * area])
    return h


def _lift_at(h, alpha_deg, speed=1.0):
    a = np.radians(alpha_deg)
    v = speed * np.array([[np.sin(a), np.cos(a), 0.0]])
    return h._lift(np.eye(3)[None], v, np.ones(1))[0], v[0] / speed


def test_lift_zero_in_plane_and_face_on():
    h = _plate()
    assert np.linalg.norm(_lift_at(h, 0)[0]) < 1e-12
    assert np.linalg.norm(_lift_at(h, 90)[0]) < 1e-12


def test_lift_magnitude_follows_sin_2alpha_and_is_perpendicular():
    h = _plate(cl=1.1, area=0.01)
    mags = {}
    for alpha in (15, 30, 45, 60, 75):
        L, v_hat = _lift_at(h, alpha)
        assert abs(L @ v_hat) < 1e-12                       # perpendicular to the flow
        expect = 0.5 * 1000.0 * 1.1 * np.sin(2 * np.radians(alpha)) * 0.01
        assert np.linalg.norm(L) == pytest.approx(expect, rel=1e-12)
        mags[alpha] = np.linalg.norm(L)
    assert max(mags, key=mags.get) == 45


def test_lift_opposes_normal_motion_and_scales_with_speed_squared():
    h = _plate()
    L_pos, _ = _lift_at(h, 30)
    L_neg, _ = _lift_at(h, -30)
    assert L_pos[0] < 0 < L_neg[0]                          # pushes back on the plate
    assert L_pos[1] == pytest.approx(L_neg[1])              # in-plane part unchanged
    assert np.linalg.norm(_lift_at(h, 45, 2.0)[0]) == pytest.approx(
        4 * np.linalg.norm(_lift_at(h, 45, 1.0)[0]))


def test_lift_switch_is_exact(cfg):
    off = make_swimmer(cfg, hydro={"lift": False})
    assert not off.hydro.has_lift
    x = off.gait.x0()
    r = off.rollout(x)
    assert r["thrust_lift"] == 0.0 and r["lift_share"] == 0.0
    on = make_swimmer(cfg, hydro={"lift": True})
    assert on.hydro.has_lift
    assert on.rollout(x)["lift_share"] > 0.0


# ---------- conservation ----------
@pytest.mark.parametrize("lift", [False, True])
def test_impulse_bookkeeping_matches_momentum_change(cfg, lift):
    """Coast-down: the only horizontal external force is hydrodynamic, so the tracked
    impulse must equal the change in momentum. This is what the lift/drag thrust split
    rests on. It failed by 36 % before body_subtreemass was refreshed."""
    sw = make_swimmer(cfg, hydro={"lift": lift})
    m, d = sw.model, sw.data
    mujoco.mj_resetData(m, d)
    for _ in range(150):
        sw.hydro.apply(d)
        mujoco.mj_step(m, d)
    d.qvel[0] = 0.25
    mujoco.mj_forward(m, d)

    def momentum():
        mujoco.mj_subtreeVel(m, d)
        return m.body_subtreemass[1] * d.subtree_linvel[1].copy()

    sw.hydro.reset_impulse()
    p0 = momentum()
    for _ in range(1000):
        sw.hydro.apply(d)
        mujoco.mj_step(m, d)
    dp = momentum() - p0
    imp = sw.hydro.imp_drag + sw.hydro.imp_lift
    assert dp[0] == pytest.approx(imp[0], rel=1e-3)


def test_subtree_mass_includes_added_mass(cfg):
    m = make_swimmer(cfg).model
    assert m.body_subtreemass[1] == pytest.approx(float(np.sum(m.body_mass[1:])))


def test_added_mass_reaches_the_dynamics(tmp_path):
    """A lone box, fully submerged, no gravity: a = F / (m + m_added) exactly."""
    xml = write_mjcf(tmp_path, """<body name="b" pos="0 0 -0.5"><freejoint/>
        <geom name="hull" type="box" size="0.1 0.05 0.05" mass="1.0"/></body>""")
    m = mujoco.MjModel.from_xml_path(xml)
    m.opt.gravity[:] = 0
    cfg = {"rules": [], "default": {"cd": [0, 0, 0], "ca": 0.5, "cv": 0.0}}
    h = HydroModel(m, cfg)
    ma = 0.5 * 1000.0 * (0.2 * 0.1 * 0.1)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(50):
        d.xfrc_applied[:] = 0
        d.xfrc_applied[1, 0] = 1.0
        mujoco.mj_step(m, d)
    a = d.qvel[0] / (50 * m.opt.timestep)
    assert a == pytest.approx(1.0 / (1.0 + ma), rel=1e-6)


def test_added_mass_cancellation_acts_at_the_centre_of_mass(tmp_path):
    """Out of the water and at rest there is no fluid force, only the constant
    cancellation of the added mass's weight. It must not create a torque, even when
    the body's centre of mass is far from its geom."""
    xml = write_mjcf(tmp_path, """<body name="b" pos="0 0 2"><freejoint/>
        <inertial pos="0.08 0 0" mass="1" diaginertia="1e-3 1e-3 1e-3"/>
        <geom name="hull" type="box" size="0.1 0.05 0.05"/></body>""")
    m = mujoco.MjModel.from_xml_path(xml)
    h = HydroModel(m, {"rules": [], "default": {"cd": [1, 1, 1], "ca": 0.5, "cv": 0}})
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    h.apply(d)
    assert d.xfrc_applied[1, 2] == pytest.approx(h.ma_body[1] * h.g)
    assert np.abs(d.xfrc_applied[1, 3:]).max() < 1e-12


def test_visual_duplicates_are_not_double_counted(tmp_path):
    xml = write_mjcf(tmp_path, """<body name="b" pos="0 0 0"><freejoint/>
        <geom name="look" type="box" size="0.1 0.05 0.05" contype="0" conaffinity="0"/>
        <geom name="hull" type="box" size="0.1 0.05 0.05"/></body>
      <body name="ghost" pos="1 0 0"><freejoint/>
        <geom name="only_visual" type="box" size="0.1 0.05 0.05" contype="0" conaffinity="0"/>
      </body>""")
    m = mujoco.MjModel.from_xml_path(xml)
    h = HydroModel(m, {"rules": [], "default": {"cd": [1, 1, 1], "ca": 0.2, "cv": 0}})
    names = sorted(i["name"] for i in h.items)
    # the duplicate is skipped; a body with only visual geometry keeps it
    assert names == ["hull", "only_visual"]
    assert h.skipped_visual == ["look"]


def test_empty_model_is_harmless(tmp_path):
    xml = write_mjcf(tmp_path, """<body name="b" pos="0 0 0"><freejoint/>
        <geom name="vis_all" type="box" size="0.1 0.1 0.1"/></body>""")
    m = mujoco.MjModel.from_xml_path(xml)
    h = HydroModel(m, {"exclude": ["vis_"]})
    d = mujoco.MjData(m)
    h.apply(d)
    assert np.all(d.xfrc_applied == 0)
    assert "resistive only" in h.summary()

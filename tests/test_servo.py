# -*- coding: utf-8 -*-
"""The servo model, the ramp-in, and the per-rollout validity metrics."""
import copy

import mujoco
import numpy as np
import pytest

from conftest import make_swimmer
import view


def _vigorous(sw):
    """Everything at the top of its range: the fastest, largest stroke allowed."""
    x = sw.gait.x0()
    x[:] = 1.0
    return x


def test_a_servo_cannot_drive_its_joint_past_no_load_speed(cfg):
    """In air, with nothing else moving, a hip told to swing far reaches the servo's
    no-load speed and no more."""
    c = copy.deepcopy(cfg)
    c["hydro"]["exclude"] = [""]                       # no fluid at all
    sw = make_swimmer(c, servo={"max_speed_dps": 300})
    m, d = sw.model, sw.data
    m.opt.gravity[:] = 0
    mujoco.mj_resetData(m, d)
    d.ctrl[0] = 1.3
    peak = 0.0
    for _ in range(400):
        mujoco.mj_step(m, d)
        peak = max(peak, abs(d.actuator_velocity[0]))
    assert np.degrees(peak) == pytest.approx(300, rel=0.02)


def test_overspeed_in_water_is_never_motor_driven(cfg):
    """A joint can still be pushed past no-load speed by water or by the neighbouring
    link swinging, as a real geared servo can be back-driven. When that happens the
    motor must be braking, never driving."""
    sw = make_swimmer(cfg, servo={"max_speed_dps": 300})
    d = sw.data
    wmax = np.radians(300) * 1.02
    driving = []

    def check(k, phase):
        w = d.actuator_velocity
        shaft = d.actuator_force - sw.back_emf * w
        fast = np.abs(w) > wmax
        driving.extend(np.flatnonzero(fast & (np.sign(shaft) == np.sign(w))))
        return True

    r = sw.rollout(_vigorous(sw), on_step=check)
    assert r["ok"]
    assert driving == []


def test_servo_model_is_stable_at_the_default_timestep(cfg):
    """Regression: rewriting torque limits from joint speed every step is explicit
    damping, and on the light flipper at dt = 2 ms it chattered at 3000 deg/s. The
    back-EMF-as-damping model must give the same answer at 2 ms and at 0.5 ms."""
    speeds = []
    for dt in (0.002, 0.0005):
        c = copy.deepcopy(cfg)
        c["timestep"] = dt
        sw = make_swimmer(c)
        r = sw.rollout(sw.gait.expand(view.demo_params(sw.gait)))
        assert r["peak_joint_speed"] <= 400 * 1.02
        speeds.append(r["speed"])
    assert speeds[0] == pytest.approx(speeds[1], rel=0.05, abs=2e-3)


def test_servo_limit_adds_back_emf_and_can_be_switched_off(cfg):
    on = make_swimmer(cfg, servo={"max_speed_dps": 400})
    off = make_swimmer(cfg, servo={"max_speed_dps": None})
    dofs = on.act_dofs
    added = on.model.dof_damping[dofs] - off.model.dof_damping[dofs]
    assert np.allclose(added, 3.0 / np.radians(400))        # stall torque / no-load speed
    assert np.all(off.back_emf == 0)
    assert "no speed limit" in off.info()


def test_ramp_in_is_simulated_but_not_scored(cfg):
    sw = make_swimmer(cfg)
    phases = {}

    def count(k, phase):
        phases[phase] = phases.get(phase, 0) + 1

    x = sw.gait.x0()
    r = sw.rollout(x, on_step=count)
    dt = sw.model.opt.timestep
    assert phases["ramp"] == round(sw.gait.ramp_t / dt)
    assert r["duration"] == pytest.approx(phases["run"] * dt)
    assert r["duration"] == pytest.approx(sw.window(sw.gait.decode(x)), abs=dt)


def test_validity_metrics_are_reported(cfg):
    sw = make_swimmer(cfg)
    r = sw.rollout(_vigorous(sw))
    for key in ("peak_joint_speed", "torque_sat", "surfacing"):
        assert key in r
    assert 0.0 <= r["torque_sat"] <= 1.0 and 0.0 <= r["surfacing"] <= 1.0
    d = sw.diverged()
    assert d["peak_joint_speed"] == d["torque_sat"] == d["surfacing"] == 0.0

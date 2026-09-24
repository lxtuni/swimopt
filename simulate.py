# -*- coding: utf-8 -*-
"""
Rollout and scoring -- wires the model, the hydrodynamics and the gait together and
turns one parameter vector into one number.

The optimizer touches the simulation only through `rollout(x)`, so it never needs to
know anything about the robot. There is exactly one simulation loop in the project:
the live viewer renders through its `on_step` callback instead of keeping a copy.
"""
import json
import math
import os

import numpy as np
import mujoco

from hydro import HydroModel
from gait import SineGait


class Swimmer:
    def __init__(self, xml_path, cfg):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        if "timestep" in cfg:
            self.model.opt.timestep = float(cfg["timestep"])
        self.hydro = HydroModel(self.model, cfg.get("hydro", {}))
        self.gait = SineGait(self.model, cfg.get("gait", {}))
        self.data = mujoco.MjData(self.model)
        self.T = float(cfg.get("sim_time", 12.0))
        self.settle = float(cfg.get("settle_time", 1.0))    # let the float settle first
        # Round the scored window up to whole strokes; see window().
        self.whole_strokes = bool(cfg.get("whole_strokes", True))
        self.vmax_guard = float(cfg.get("vmax_guard", 5.0))
        self.trunk = cfg.get("trunk_body", None)
        self.trunk_id = (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.trunk)
                         if self.trunk else 1)
        self.w_yaw = float(cfg.get("w_yaw", 0.3))           # heading penalty weight
        self.w_energy = float(cfg.get("w_energy", 0.0))     # 0 means speed only
        # Attitude penalty. Without it nothing stops the optimizer from rolling the
        # hull over to get a faster stroke, which it will do given the chance.
        self.w_attitude = float(cfg.get("w_attitude", 0.05))
        self.body_len = float(cfg.get("body_length", 0.0))  # metres, for body-lengths/s
        if self.body_len <= 0:
            self.body_len = self._estimate_body_length()
        self.free_base = any(self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                             for j in range(self.model.njnt))
        self._settled = None                                # MjData after settling

    def _estimate_body_length(self):
        """Span along world x of every robot geom's bounding sphere, initial pose."""
        d = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, d)
        lo, hi = np.inf, -np.inf
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == 0:              # water plane and scenery
                continue
            x = float(d.geom_xpos[g][0])
            r = float(self.model.geom_rbound[g])
            lo = min(lo, x - r)
            hi = max(hi, x + r)
        return max(hi - lo, 1e-3) if hi > lo else 1e-3

    # ---------- settle cache ----------
    def _restore_settled(self, on_step=None):
        """Put self.data into the post-settle state, simulating it only once.

        Settling runs with every actuator at zero, so it is identical for every
        candidate, and re-simulating it cost about a tenth of every rollout. The
        snapshot is a complete MjData copy, so a restored rollout is bit-identical to
        a freshly settled one. It assumes the model is not modified after the first
        rollout; call invalidate_settle() if you do.
        """
        m, d = self.model, self.data
        if self._settled is not None:
            mujoco.mj_copyData(d, m, self._settled)
            return True
        mujoco.mj_resetData(m, d)
        d.ctrl[:] = 0
        for k in range(int(self.settle / m.opt.timestep)):
            self.hydro.apply(d)
            mujoco.mj_step(m, d)
            if on_step is not None and on_step(k, "settle") is False:
                return False                   # aborted: never cache a partial settle
        self._settled = mujoco.MjData(m)
        mujoco.mj_copyData(self._settled, m, d)
        return True

    def invalidate_settle(self):
        self._settled = None

    def window(self, p):
        """Scored duration: sim_time, rounded up to a whole number of strokes.

        Speed comes from end-point displacement, so a window that ends mid-stroke is
        biased by up to half a stroke's travel. At 0.25 Hz an 8 s window holds just two
        strokes and that bias is large. Rounding up removes it and never measures less
        than sim_time.
        """
        if not self.whole_strokes:
            return self.T
        per = self.gait.period(p)
        return max(math.ceil(self.T / per - 1e-9), 1) * per

    # ---------- one rollout ----------
    def rollout(self, x, record=False, on_step=None):
        """Simulate one candidate and score it.

        on_step(k, phase) is called after every simulation step, phase being "settle"
        or "run". Returning False aborts the rollout, which then returns None.
        """
        x = self.gait.expand(x)          # accepts a reduced vector when params are frozen
        m, d = self.model, self.data
        p = self.gait.decode(x)
        if not self._restore_settled(on_step):
            return None
        dt = m.opt.timestep
        duration = self.window(p)
        n_steps = int(round(duration / dt))
        duration = n_steps * dt
        traj = []

        tid = self.trunk_id
        p0 = d.xpos[tid].copy()
        yaw0 = self._yaw()
        self.hydro.reset_impulse()      # only count thrust from the scored window
        guard_q = self.vmax_guard * 20
        guard_v2 = self.vmax_guard ** 2
        # Views into MjData: they track the live values across steps.
        xmat, cvel = d.xmat[tid], d.cvel[tid]
        af, av, qvel = d.actuator_force, d.actuator_velocity, d.qvel
        energy = 0.0
        pitch_max = roll_max = pitch_sq = roll_sq = 0.0
        for k in range(n_steps):
            d.ctrl[:] = self.gait.ctrl(p, k * dt)
            self.hydro.apply(d)
            mujoco.mj_step(m, d)
            # "not <=" also catches NaN, which compares false against everything
            if not (np.abs(qvel).max() <= guard_q) or \
                    cvel[3] * cvel[3] + cvel[4] * cvel[4] + cvel[5] * cvel[5] > guard_v2:
                return self.diverged(traj)
            energy += float(np.abs(af * av).sum()) * dt
            pitch = abs(math.asin(min(1.0, max(-1.0, -xmat[6]))))
            roll = abs(math.atan2(xmat[7], xmat[8]))
            if pitch > pitch_max:
                pitch_max = pitch
            if roll > roll_max:
                roll_max = roll
            pitch_sq += pitch * pitch
            roll_sq += roll * roll
            if record and k % 50 == 0:
                traj.append([k * dt, *d.xpos[tid], np.degrees(self._yaw() - yaw0)])
            if on_step is not None and on_step(k, "run") is False:
                return None

        disp = d.xpos[tid] - p0
        yaw_drift = abs(np.degrees(self._wrap(self._yaw() - yaw0)))
        # net displacement projected on the initial heading, so circling scores badly
        fwd = self._forward_dir(yaw0)
        dist = float(disp[:2] @ fwd)
        n = max(n_steps, 1)
        res = self.score(dist=dist, yaw_drift=yaw_drift, energy=energy,
                         roll_rms=math.sqrt(roll_sq / n), pitch_rms=math.sqrt(pitch_sq / n),
                         roll_max=roll_max, pitch_max=pitch_max, traj=traj,
                         duration=duration)
        res.update(self.thrust_split(fwd))
        return res

    def thrust_split(self, fwd):
        """Signed forward impulse from the lift and from the resistive terms.

        This is the quantitative test for whether a gait is lift-based or drag-based,
        rather than judging it by eye. thrust_* are impulses along the heading in
        N*s, positive driving the robot forwards. lift_share is only their magnitude
        ratio, so read it together with the signs. The decomposition is checked
        against momentum conservation by the test suite.
        """
        drag = float(self.hydro.imp_drag[:2] @ fwd)
        lift = float(self.hydro.imp_lift[:2] @ fwd)
        total = abs(drag) + abs(lift)
        return dict(thrust_drag=drag, thrust_lift=lift,
                    lift_share=float(abs(lift) / total) if total > 1e-12 else 0.0)

    # ---------- scoring ----------
    def score(self, dist, yaw_drift, energy, roll_rms, pitch_rms,
              roll_max=0.0, pitch_max=0.0, traj=None, duration=None):
        """Turn one rollout's raw measurements into a fitness and a result dict.

        `dist` is metres along the initial heading, `yaw_drift` degrees, `energy`
        joules, the attitude terms radians, `duration` seconds (default sim_time).
        """
        T = self.T if duration is None else float(duration)
        speed = dist / T
        power = energy / T                       # mean watts, independent of sim length
        yaw_rate = np.radians(yaw_drift) / T
        attitude = float(roll_rms + pitch_rms)   # radians, RMS over the rollout
        fit = (speed
               - self.w_yaw * yaw_rate
               - self.w_energy * power
               - self.w_attitude * attitude)
        return dict(ok=True, fitness=float(fit), dist=float(dist), speed=float(speed),
                    bl_s=float(speed / self.body_len),
                    yaw=float(yaw_drift),
                    pitch_amp=float(np.degrees(pitch_max)),
                    roll_amp=float(np.degrees(roll_max)),
                    pitch_rms=float(np.degrees(pitch_rms)),
                    roll_rms=float(np.degrees(roll_rms)),
                    attitude=attitude,
                    energy=float(energy), power=float(power), duration=T,
                    thrust_drag=0.0, thrust_lift=0.0, lift_share=0.0,
                    traj=traj if traj is not None else [])

    def diverged(self, traj=None):
        """The result of a rollout that hit the speed guard."""
        return dict(ok=False, fitness=-1e3, dist=0.0, speed=0.0, bl_s=0.0, yaw=0.0,
                    pitch_amp=0.0, roll_amp=0.0, pitch_rms=0.0, roll_rms=0.0,
                    attitude=0.0, energy=0.0, power=0.0, duration=0.0,
                    thrust_drag=0.0, thrust_lift=0.0, lift_share=0.0,
                    traj=traj if traj is not None else [])

    def evaluate(self, x):
        """CMA-ES minimizes, so return the negated fitness."""
        return -self.rollout(x)["fitness"]

    # ---------- helpers ----------
    def _yaw(self):
        R = self.data.xmat[self.trunk_id]
        return math.atan2(R[3], R[0])

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _forward_dir(yaw0):
        return np.array([np.cos(yaw0), np.sin(yaw0)])

    def info(self):
        lines = [
            self.hydro.summary(),
            self.gait.info(),
            self.gait.opt_summary(),
            f"[sim] dt={self.model.opt.timestep*1000:.1f}ms duration={self.T}s"
            + (" (rounded up to whole strokes)" if self.whole_strokes else ""),
            f"[objective] fitness = speed - {self.w_yaw} x yaw rate "
            f"- {self.w_energy} x mean power - {self.w_attitude} x attitude",
        ]
        if not self.free_base:
            lines.append("[!] this model has NO free joint: the robot is welded to the "
                         "world and cannot swim. Re-import it with import_model.py.")
        return "\n".join(lines)


def record_best(result, x, gait):
    """The part of a rollout result that gets saved to best.json.

    It stores the FULL parameter vector, frozen entries included, so replaying a
    result never depends on which parameters the current config happens to freeze.
    Storing only the optimized subset let a later change of preset silently replay a
    different gait.
    """
    return dict(fitness=result["fitness"], speed=result["speed"], bl_s=result["bl_s"],
                yaw=result["yaw"], roll=result["roll_amp"], pitch=result["pitch_amp"],
                power=result["power"], duration=result["duration"],
                thrust_lift=result["thrust_lift"], thrust_drag=result["thrust_drag"],
                x=list(map(float, gait.expand(x))), dim=int(gait.dim),
                optimized=[k for k, v in gait.opt_flags.items() if v],
                preset=gait.preset or None)


def atomic_json_dump(obj, path, **kw):
    """Write JSON so a concurrent reader never sees a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, **kw)
    os.replace(tmp, path)


def strip_jsonc(text):
    """Remove // and /* */ comments outside of strings, so config.json can be annotated."""
    out, i, n = [], 0, len(text)
    in_str = in_line = in_blk = False
    esc = False
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_line:
            if c == "\n":
                in_line = False
                out.append(c)
        elif in_blk:
            if c == "*" and nxt == "/":
                in_blk = False
                i += 1
        elif in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == "/" and nxt == "/":
                in_line = True
                i += 1
            elif c == "/" and nxt == "*":
                in_blk = True
                i += 1
            else:
                if c == '"':
                    in_str = True
                out.append(c)
        i += 1
    return "".join(out)


def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(strip_jsonc(f.read()))

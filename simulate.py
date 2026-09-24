# -*- coding: utf-8 -*-
"""
Rollout and scoring -- wires the model, the hydrodynamics and the gait together and
turns one parameter vector into one number.

The optimizer touches the simulation only through `evaluate(x)`, so it never needs to
know anything about the robot.
"""
import json

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

    def _estimate_body_length(self):
        """Span along x of every geom's bounding sphere, used to report BL/s."""
        lo, hi = np.inf, -np.inf
        for g in range(self.model.ngeom):
            x = float(self.model.geom_pos[g][0])
            r = float(self.model.geom_rbound[g])
            lo = min(lo, x - r)
            hi = max(hi, x + r)
        return max(hi - lo, 1e-3)

    # ---------- one rollout ----------
    def rollout(self, x, record=False):
        x = self.gait.expand(x)          # accepts a reduced vector when params are frozen
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)        # deterministic reset: identical x gives identical run
        p = self.gait.decode(x)
        dt = m.opt.timestep
        n_settle = int(self.settle / dt)
        n_steps = int(self.T / dt)
        traj = []

        # settling phase: no actuation, wait for the float to stabilize
        d.ctrl[:] = 0
        for _ in range(n_settle):
            self.hydro.apply(d)
            mujoco.mj_step(m, d)

        p0 = d.xpos[self.trunk_id].copy()
        yaw0 = self._yaw()
        self.hydro.reset_impulse()      # only count thrust from the scored window
        energy = 0.0
        blew_up = False
        pitch_max = roll_max = 0.0
        pitch_sq = roll_sq = 0.0
        n_att = 0
        for k in range(n_steps):
            t = k * dt
            d.ctrl[:] = self.gait.ctrl(p, t)
            self.hydro.apply(d)
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qvel)) or np.max(np.abs(d.qvel)) > self.vmax_guard * 20:
                blew_up = True
                break
            v = np.linalg.norm(d.cvel[self.trunk_id, 3:6])
            if v > self.vmax_guard:
                blew_up = True
                break
            energy += float(np.sum(np.abs(d.actuator_force * d.actuator_velocity))) * dt
            R = d.xmat[self.trunk_id].reshape(3, 3)
            pitch = abs(float(np.arcsin(np.clip(-R[2, 0], -1, 1))))
            roll = abs(float(np.arctan2(R[2, 1], R[2, 2])))
            pitch_max = max(pitch_max, pitch)
            roll_max = max(roll_max, roll)
            pitch_sq += pitch * pitch
            roll_sq += roll * roll
            n_att += 1
            if record and k % 50 == 0:
                traj.append([t, *d.xpos[self.trunk_id], np.degrees(self._yaw() - yaw0)])

        if blew_up:
            return self.diverged(traj)

        p1 = d.xpos[self.trunk_id].copy()
        disp = p1 - p0
        yaw_drift = abs(np.degrees(self._wrap(self._yaw() - yaw0)))
        # net displacement projected on the initial heading, so circling scores badly
        fwd = self._forward_dir(yaw0)
        dist = float(disp[:2] @ fwd)
        n_att = max(n_att, 1)
        res = self.score(dist=dist, yaw_drift=yaw_drift, energy=energy,
                         roll_rms=np.sqrt(roll_sq / n_att),
                         pitch_rms=np.sqrt(pitch_sq / n_att),
                         roll_max=roll_max, pitch_max=pitch_max, traj=traj)
        res.update(self.thrust_split(fwd))
        return res

    def thrust_split(self, fwd):
        """How much of the forward impulse came from lift and how much from drag.

        This is the quantitative test for whether a gait is lift-based or
        drag-based, rather than judging it by eye from the animation.
        """
        drag = float(self.hydro.imp_drag[:2] @ fwd)
        lift = float(self.hydro.imp_lift[:2] @ fwd)
        total = abs(drag) + abs(lift)
        # thrust_* are signed impulses along the heading, in N*s: positive drives the
        # robot forwards. lift_share is their magnitude ratio only, so read it together
        # with the signs -- a large share can mean lift is doing the pushing or the
        # holding back. The decomposition is checked against momentum conservation in
        # a coast-down, where the two agree to better than 0.01 %.
        return dict(thrust_drag=drag, thrust_lift=lift,
                    lift_share=float(abs(lift) / total) if total > 1e-12 else 0.0)

    # ---------- scoring ----------
    # One place computes the objective. optimize_view.py runs its own render-aware
    # loop but calls straight into here, so the watched and unwatched searches can
    # never drift apart on what a gait is worth.
    def score(self, dist, yaw_drift, energy, roll_rms, pitch_rms,
              roll_max=0.0, pitch_max=0.0, traj=None):
        """Turn one rollout's raw measurements into a fitness and a result dict.

        `dist` is metres along the initial heading, `yaw_drift` degrees, `energy`
        joules, and the attitude terms radians.
        """
        speed = dist / self.T
        power = energy / self.T                  # mean watts, independent of sim length
        yaw_rate = np.radians(yaw_drift) / self.T
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
                    energy=float(energy), power=float(power),
                    thrust_drag=0.0, thrust_lift=0.0, lift_share=0.0,
                    traj=traj if traj is not None else [])

    def diverged(self, traj=None):
        """The result of a rollout that hit the speed guard."""
        return dict(ok=False, fitness=-1e3, dist=0.0, speed=0.0, bl_s=0.0, yaw=0.0,
                    pitch_amp=0.0, roll_amp=0.0, pitch_rms=0.0, roll_rms=0.0,
                    attitude=0.0, energy=0.0, power=0.0,
                    thrust_drag=0.0, thrust_lift=0.0, lift_share=0.0,
                    traj=traj if traj is not None else [])

    def evaluate(self, x):
        """CMA-ES minimizes, so return the negated fitness."""
        return -self.rollout(x)["fitness"]

    # ---------- helpers ----------
    def _yaw(self):
        R = self.data.xmat[self.trunk_id].reshape(3, 3)
        return np.arctan2(R[1, 0], R[0, 0])

    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _forward_dir(yaw0):
        return np.array([np.cos(yaw0), np.sin(yaw0)])

    def info(self):
        return "\n".join([
            self.hydro.summary(),
            self.gait.info(),
            self.gait.opt_summary(),
            f"[sim] dt={self.model.opt.timestep*1000:.1f}ms duration={self.T}s",
            f"[objective] fitness = speed - {self.w_yaw} x yaw rate "
            f"- {self.w_energy} x mean power - {self.w_attitude} x attitude",
        ])


def record_best(result, x):
    """The part of a rollout result that gets saved to best.json.

    Both optimizers write it, so keeping one definition stops the two files from
    growing different fields.
    """
    return dict(fitness=result["fitness"], speed=result["speed"], bl_s=result["bl_s"],
                yaw=result["yaw"], roll=result["roll_amp"], pitch=result["pitch_amp"],
                power=result["power"], x=list(map(float, x)))


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

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

# RMS over the scored window, in degrees. See Swimmer.__init__ and score().
DEFAULT_LIMITS = {"heading_deg": 10.0, "roll_deg": 10.0, "pitch_deg": 15.0}
# Fitness bands. Every feasible gait scores above INFEASIBLE, every infeasible one
# between INFEASIBLE and DIVERGED, and a rollout that blew up at DIVERGED. The bands
# are far enough apart that no gait can cross one.
INFEASIBLE = -1e3
DIVERGED = -1e6


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
        # Straight and level are limits, not weights. An additive weight only works at
        # the speed scale it was tuned for: once the legs could really move, speed
        # swamped the old w_yaw/w_attitude terms and the optimizer happily rolled and
        # veered. Within a limit only speed counts; beyond it the score drops by one
        # body length per second for every 100 % of excess.
        lim = dict(DEFAULT_LIMITS, **cfg.get("limits", {}))
        self.limits = {k: float(lim[f"{k}_deg"]) for k in ("heading", "roll", "pitch")}
        # Optional: the largest share of the scored time a limb may spend breaking the
        # surface. The model has no free-surface physics, so a gait that rows across
        # the surface is outside what it can predict. None = not enforced.
        surf = lim.get("surfacing")
        self.surfacing_limit = None if surf is None else float(surf)
        self.w_energy = float(cfg.get("w_energy", 0.0))     # per watt, in BL/s
        self.legacy_weights = [k for k in ("w_yaw", "w_attitude") if k in cfg]
        # What to optimize. "speed": fastest within the limits. "power": least mean
        # power that still reaches min_speed within the limits -- one point on the
        # speed-power trade-off per min_speed.
        obj = cfg.get("objective") or {}
        self.mode = obj.get("mode", "speed")
        if self.mode not in ("speed", "power"):
            raise ValueError(f"objective.mode must be 'speed' or 'power', not {self.mode!r}")
        self.min_speed = float(obj.get("min_speed", 0.0))
        self.body_len = float(cfg.get("body_length", 0.0))  # metres, for body-lengths/s
        if self.body_len <= 0:
            self.body_len = self._estimate_body_length()
        self.free_base = any(self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                             for j in range(self.model.njnt))
        self._settled = None                                # MjData after settling
        self._setup_actuation(cfg.get("servo", {}))

    def _setup_actuation(self, servo_cfg):
        m = self.model
        # Servo speed limit, as a per-step bound on how far the command may move.
        rate = servo_cfg.get("max_speed_dps", 400)
        self.servo_speed = float(rate) if rate else None
        limited = m.actuator_ctrllimited.astype(bool)
        self.ctrl_lo = np.where(limited, m.actuator_ctrlrange[:, 0], -np.inf)
        self.ctrl_hi = np.where(limited, m.actuator_ctrlrange[:, 1], np.inf)
        # For the per-rollout checks: which dofs the actuators drive, and their
        # torque limits.
        joint_act = m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT
        self.act_dofs = m.jnt_dofadr[m.actuator_trnid[joint_act, 0]]
        flim = m.actuator_forcelimited.astype(bool)
        self.tau_stall = np.where(flim, np.abs(m.actuator_forcerange).max(1), np.inf)
        self.force_hi = self.tau_stall * (1 - 1e-9)
        self._add_back_emf(flim & joint_act)

    def _add_back_emf(self, which):
        """Give each servo a DC motor's torque-speed curve.

        A servo motor has a stall torque ts and a no-load speed wmax, and its torque
        falls linearly between them:  tau = ts * v - (ts / wmax) * w,  |v| <= 1.
        The second term is back-EMF, and it is just viscous damping with coefficient
        b = ts / wmax. So: the position actuator, clipped at +/-ts, supplies the first
        term, and b is added to the joint's damping, which MuJoCo integrates
        implicitly. The joint then cannot outrun wmax under its own power.

        Two simpler versions failed, and the tests keep them failed:
        - limiting only the command: when drag held a hip back, the servo fell behind
          and then whipped the leg forward at 1500 deg/s to catch up;
        - rewriting the torque limits from the joint speed each step: that is explicit
          damping, and on a 2e-4 kg*m^2 flipper at dt = 2 ms it went unstable and
          chattered at 3000 deg/s.
        """
        self.back_emf = np.zeros(self.model.nu)
        if not self.servo_speed:
            return
        wmax = math.radians(self.servo_speed)
        for a in np.flatnonzero(which):
            b = self.tau_stall[a] / wmax
            dof = self.model.jnt_dofadr[self.model.actuator_trnid[a, 0]]
            self.model.dof_damping[dof] += b
            self.back_emf[a] = b

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
    def _restore_settled(self, tick):
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
        for _ in range(int(self.settle / m.opt.timestep)):
            self.hydro.apply(d)
            mujoco.mj_step(m, d)
            if not tick("settle"):
                return False                   # aborted: never cache a partial settle
        self._settled = mujoco.MjData(m)
        mujoco.mj_copyData(self._settled, m, d)
        return True

    def _command(self, p, t, u_prev):
        """The position each servo is told to go to at time t, within its range.

        The servo's speed limit lives in the motor model (see _add_back_emf), not in
        the command: a real servo is simply told where to go and gets there as fast as
        its motor allows.
        """
        return np.clip(self.gait.ctrl(p, t), self.ctrl_lo, self.ctrl_hi)

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
    def rollout(self, x, record=False, on_step=None, duration=None):
        """Simulate one candidate and score it.

        Three phases: "settle" (actuators idle; simulated once, then restored from a
        cache), "ramp" (the gait fades in over gait.ramp_t; simulated but not scored,
        because a speed averaged over the fade-in understates the steady gait and does
        so differently at different frequencies), and "run" (scored).

        on_step(k, phase) is called after every simulation step, k counting steps
        across all phases of this call; returning False aborts the rollout, which then
        returns None. Playback and the live viewer render through it, so there is one
        simulation loop in the project. `duration` overrides the scored window.
        """
        x = self.gait.expand(x)          # accepts a reduced vector when params are frozen
        m, d = self.model, self.data
        p = self.gait.decode(x)
        count = [0]

        def tick(phase):
            if on_step is None:
                return True
            k = count[0]
            count[0] += 1
            return on_step(k, phase) is not False

        if not self._restore_settled(tick):
            return None
        dt = m.opt.timestep
        n_ramp = int(round(self.gait.ramp_t / dt)) if self.gait.ramp_t > 0 else 0
        span = self.window(p) if duration is None else float(duration)
        n_steps = max(int(round(span / dt)), 1)
        span = n_steps * dt
        traj = []

        tid = self.trunk_id
        guard_q = self.vmax_guard * 20
        guard_v2 = self.vmax_guard ** 2
        # Views into MjData: they track the live values across steps.
        xmat, cvel = d.xmat[tid], d.cvel[tid]
        af, av, qvel = d.actuator_force, d.actuator_velocity, d.qvel
        u = d.ctrl.copy()

        def blew_up():
            # "not <=" also catches NaN, which compares false against everything
            return (not (np.abs(qvel).max() <= guard_q)
                    or cvel[3] * cvel[3] + cvel[4] * cvel[4] + cvel[5] * cvel[5] > guard_v2)

        for k in range(n_ramp):
            u = self._command(p, k * dt, u)
            d.ctrl[:] = u
            self.hydro.apply(d)
            mujoco.mj_step(m, d)
            if blew_up():
                return self.diverged(traj)
            if not tick("ramp"):
                return None

        p0 = d.xpos[tid].copy()
        yaw0 = self._yaw()
        self.hydro.reset_impulse()      # only count thrust from the scored window
        energy = 0.0
        pitch_max = roll_max = pitch_sq = roll_sq = head_sq = 0.0
        peak_w = 0.0
        n_sat = n_surf = 0
        limbs = self.hydro.is_limb
        two_pi = 2.0 * math.pi
        for k in range(n_steps):
            t = (n_ramp + k) * dt
            u = self._command(p, t, u)
            d.ctrl[:] = u
            self.hydro.apply(d)
            mujoco.mj_step(m, d)
            if blew_up():
                return self.diverged(traj)
            # power at the motor shaft: actuator torque less the back-EMF drag
            energy += float(np.abs((af - self.back_emf * av) * av).sum()) * dt
            pitch = abs(math.asin(min(1.0, max(-1.0, -xmat[6]))))
            roll = abs(math.atan2(xmat[7], xmat[8]))
            if pitch > pitch_max:
                pitch_max = pitch
            if roll > roll_max:
                roll_max = roll
            pitch_sq += pitch * pitch
            roll_sq += roll * roll
            # heading deviation from the start, wrapped to (-pi, pi]
            dh = (math.atan2(xmat[3], xmat[0]) - yaw0 + math.pi) % two_pi - math.pi
            head_sq += dh * dh
            # validity: joint speed the servos must deliver, torque saturation, and
            # limbs breaking the surface (which this model has no physics for)
            w = float(np.abs(qvel[self.act_dofs]).max()) if self.act_dofs.size else 0.0
            if w > peak_w:
                peak_w = w
            if (np.abs(af) >= self.force_hi).any():
                n_sat += 1
            if limbs.size and (self.hydro.f_last[limbs] < 0.5).any():
                n_surf += 1
            if record and k % 50 == 0:
                traj.append([t, *d.xpos[tid], np.degrees(self._yaw() - yaw0)])
            if not tick("run"):
                return None

        disp = d.xpos[tid] - p0
        yaw_drift = abs(np.degrees(self._wrap(self._yaw() - yaw0)))
        # net displacement projected on the initial heading, so circling scores badly
        fwd = self._forward_dir(yaw0)
        dist = float(disp[:2] @ fwd)
        res = self.score(dist=dist, yaw_drift=yaw_drift, energy=energy,
                         roll_rms=math.sqrt(roll_sq / n_steps),
                         pitch_rms=math.sqrt(pitch_sq / n_steps),
                         heading_rms=math.sqrt(head_sq / n_steps),
                         roll_max=roll_max, pitch_max=pitch_max, traj=traj,
                         duration=span, surfacing=n_surf / n_steps)
        res.update(self.thrust_split(fwd))
        res.update(peak_joint_speed=math.degrees(peak_w), torque_sat=n_sat / n_steps,
                   surfacing=n_surf / n_steps)
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
    def score(self, dist, yaw_drift, energy, roll_rms, pitch_rms, heading_rms=0.0,
              roll_max=0.0, pitch_max=0.0, traj=None, duration=None, surfacing=0.0):
        """Turn one rollout's raw measurements into a fitness and a result dict.

Feasibility comes first. A gait is feasible when its RMS heading, roll and pitch
        are within their limits (and, for the power goal, it reaches min_speed). Then:

          feasible, speed goal:  fitness = speed in body lengths/s - w_energy * power
          feasible, power goal:  fitness = -mean power in watts
          infeasible:            fitness = INFEASIBLE - total relative excess

        so no infeasible gait can outscore a feasible one, however fast it is, and
        among infeasible ones the search is pulled toward feasibility. An earlier
        version subtracted 1 BL/s per 100 % of excess; that assumed nothing swims
        faster than 1 BL/s, and unconstrained gaits here reach 5.

        `dist` is metres along the initial heading, `yaw_drift` degrees, `energy`
        joules, the RMS terms radians, `duration` seconds (default sim_time).
        """
        T = self.T if duration is None else float(duration)
        speed = dist / T
        bl_s = speed / self.body_len
        power = energy / T                       # mean watts, independent of sim length
        rms = {"heading": math.degrees(heading_rms), "roll": math.degrees(roll_rms),
               "pitch": math.degrees(pitch_rms)}
        excess = {k: max(0.0, rms[k] - self.limits[k]) / self.limits[k] for k in rms}
        penalty = sum(excess.values())
        if self.surfacing_limit is not None:
            penalty += (max(0.0, surfacing - self.surfacing_limit)
                        / max(self.surfacing_limit, 0.01))
        if self.mode == "power":
            if self.min_speed > 0:
                penalty += max(0.0, self.min_speed - speed) / self.min_speed
            objective = -power
        else:
            objective = bl_s - self.w_energy * power
        fit = objective if penalty == 0.0 else INFEASIBLE - penalty
        return dict(ok=True, fitness=float(fit), dist=float(dist), speed=float(speed),
                    bl_s=float(bl_s), yaw=float(yaw_drift),
                    heading_rms=rms["heading"], roll_rms=rms["roll"],
                    pitch_rms=rms["pitch"],
                    pitch_amp=float(np.degrees(pitch_max)),
                    roll_amp=float(np.degrees(roll_max)),
                    penalty=float(penalty), feasible=penalty == 0.0,
                    energy=float(energy), power=float(power), duration=T,
                    thrust_drag=0.0, thrust_lift=0.0, lift_share=0.0,
                    traj=traj if traj is not None else [])

    def diverged(self, traj=None):
        """The result of a rollout that hit the speed guard."""
        return dict(ok=False, fitness=DIVERGED, dist=0.0, speed=0.0, bl_s=0.0, yaw=0.0,
                    heading_rms=0.0, roll_rms=0.0, pitch_rms=0.0,
                    pitch_amp=0.0, roll_amp=0.0, penalty=0.0, feasible=False,
                    energy=0.0, power=0.0, duration=0.0,
                    thrust_drag=0.0, thrust_lift=0.0, lift_share=0.0,
                    peak_joint_speed=0.0, torque_sat=0.0, surfacing=0.0,
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
            f"[sim] dt={self.model.opt.timestep*1000:.1f}ms, scored {self.T}s"
            + (" rounded up to whole strokes" if self.whole_strokes else "")
            + f", after a {self.gait.ramp_t:g}s unscored ramp-in",
            "[servo] " + (f"max {self.servo_speed:g} deg/s" if self.servo_speed
                          else "no speed limit (servos can follow any command)"),
            "[objective] " + (
                f"least mean power at speed >= {self.min_speed:g} m/s"
                if self.mode == "power" else
                f"fastest (body lengths/s - {self.w_energy} x mean power)")
            + f", within RMS heading {self.limits['heading']:g}, roll "
              f"{self.limits['roll']:g}, pitch {self.limits['pitch']:g} deg",
        ]
        if self.legacy_weights:
            lines.append(f"[!] {', '.join(self.legacy_weights)} in the config are no "
                         f"longer used; straightness and level are set by 'limits'")
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
                heading_rms=result["heading_rms"], roll_rms=result["roll_rms"],
                pitch_rms=result["pitch_rms"], feasible=result["feasible"],
                power=result["power"], duration=result["duration"],
                thrust_lift=result["thrust_lift"], thrust_drag=result["thrust_drag"],
                peak_joint_speed=result["peak_joint_speed"],
                torque_sat=result["torque_sat"], surfacing=result["surfacing"],
                x=list(map(float, gait.expand(x))), dim=int(gait.dim),
                optimized=[k for k, v in gait.opt_flags.items() if v],
                family=gait.family)


def swimmer_for_result(cfg, best):
    """A Swimmer that decodes `best` the way it was found.

    The same numbers mean a different gait under a different gait family, so a result
    is always replayed with the family recorded in it, whatever the config now says.
    Older results that predate families carry a "preset" key instead, and were
    free-phase unless phase was frozen.
    """
    c = json.loads(json.dumps(cfg))                    # deep copy, config is JSON
    g = c.setdefault("gait", {})
    if "family" in best:
        fam = best["family"]
    else:
        # Written before families existed: a preset only ever applied with phase
        # frozen, and otherwise every phase was free.
        frozen = "phase" not in best.get("optimized", ["phase"])
        fam = best.get("preset") if frozen else None
    g["family"] = fam
    g.pop("preset_phase", None)
    return Swimmer(c["model"], c)


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

# -*- coding: utf-8 -*-
"""
Layer 3, control -- robot-agnostic gait parameterization.

Scans every actuator in the model and drives it with a truncated Fourier series:

    q_i(t) = offset_i + sum_{h=1..H} amp_i^h * sin(h * theta_i(t) + phase_i^h)

where theta_i is the stroke angle: 0..pi over the power stroke, pi..2pi over the
recovery, with the power stroke taking a fraction `duty` of the period.

The second harmonic is invariant under a phase shift of pi, which makes it the natural
way to express flipper feathering: spread on the power stroke, furl on recovery. It is
not required for net thrust here. With several independently phased joints per leg a
single harmonic already sweeps the leg non-reciprocally (0.16 m/s on toy_quad with a
pure sinusoid, 0.30 m/s with the duty warp); an earlier comment claiming otherwise was
wrong and had never been tested.

Two ways to parameterize phase:

  free    every actuator has its own phase per harmonic. The search can find any
          coordination, including ones no textbook names. 35 dims on toy_quad.
  family  the legs keep a textbook timing -- trot, pace, bound, walk, pronk -- as a
          time delay of each whole leg, and only the phase of each joint *type* (hip,
          knee, flipper) is searched, shared by all legs. 17 dims on toy_quad. This is
          how gait families are compared fairly: each at its own best stroke.

Legs and joint types come from the model's geometry, not from actuator names: a leg is
the chain of bodies hanging off the floating trunk, front/back and left/right from
where it attaches, and a joint's type is its depth down that chain.

Actuators can share amp/offset through `groups`, which lowers the search dimension:

    groups = [{"match": "1.1", "share": ["amp", "offset"]},
              {"match": "2.1", "share": ["amp", "offset"]}]
"""
import numpy as np
import mujoco

LEGS = ("FL", "FR", "BL", "BR")

# Each leg's delay, as a fraction of the stroke period, behind the front-left leg.
FAMILIES = {
    "diag":    {"FL": 0.0, "FR": 0.5, "BL": 0.5, "BR": 0.0},    # trot
    "lr":      {"FL": 0.0, "FR": 0.5, "BL": 0.0, "BR": 0.5},    # pace
    "fb":      {"FL": 0.0, "FR": 0.0, "BL": 0.5, "BR": 0.5},    # bound
    "wave":    {"FL": 0.0, "FR": 0.5, "BL": 0.75, "BR": 0.25},  # lateral-sequence walk
    "inphase": {"FL": 0.0, "FR": 0.0, "BL": 0.0, "BR": 0.0},    # pronk
}
ALIASES = {"trot": "diag", "pace": "lr", "bound": "fb", "walk": "wave", "pronk": "inphase"}
COMMON_NAMES = {"diag": "trot", "lr": "pace", "fb": "bound", "wave": "walk",
                "inphase": "pronk"}


def family_key(name):
    """Canonical family key for a name or alias, or None for free phases."""
    if name in (None, "", "free"):
        return None
    k = ALIASES.get(name, name)
    if k not in FAMILIES:
        raise ValueError(f"unknown gait family {name!r}; choose from "
                         f"{sorted(FAMILIES) + sorted(ALIASES)}")
    return k


def limb_layout(model):
    """Which leg each actuator belongs to, and how deep down that leg its joint is.

    A leg is the chain of bodies below one child of the floating trunk. Front/back and
    left/right come from where that child attaches to the trunk (MuJoCo: x forward,
    y left). Depth 0 is the joint nearest the trunk. Returns (legs, depth), with None
    for an actuator that cannot be placed; names containing FL/FR/BL/BR are used as a
    fallback when the geometry is ambiguous.
    """
    parent = model.body_parentid
    free = [model.jnt_bodyid[j] for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    root = int(free[0]) if free else 1
    legs, depth = [], []
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        leg, dpt = None, None
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            b = int(model.jnt_bodyid[model.actuator_trnid[a, 0]])
            dpt = 0
            while b not in (0, root) and parent[b] != root:
                b = int(parent[b])
                dpt += 1
            if b not in (0, root):
                x, y = model.body_pos[b][:2]
                tol = 1e-6
                if abs(x) > tol and abs(y) > tol:
                    leg = ("F" if x > 0 else "B") + ("L" if y > 0 else "R")
        if leg is None:
            leg = next((L for L in LEGS if L in name), None)
        legs.append(leg)
        depth.append(dpt if dpt is not None else 0)
    return legs, np.array(depth, dtype=int)


def _circ(a):
    """Signed circular difference of cycle fractions, in [-0.5, 0.5)."""
    return (np.asarray(a) + 0.5) % 1.0 - 0.5


class SineGait:
    def __init__(self, model, cfg=None):
        cfg = cfg or {}
        self.m = model
        self.names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act{i}"
                      for i in range(model.nu)]
        self.n = model.nu
        self.freq_range = cfg.get("freq_range", [0.2, 3.0])
        self.amp_range = cfg.get("amp_range", [0.0, 0.9])
        self.off_range = cfg.get("offset_range", [-0.5, 0.5])
        # Duty cycle of the power stroke. The paddling literature compares 25/33/50 %:
        # a fast power stroke with slow recovery, versus the reverse.
        self.duty_range = cfg.get("duty_range", [0.25, 0.75])
        self.ramp_t = cfg.get("ramp_t", 1.5)
        self.groups = cfg.get("groups", [])
        self.H = int(cfg.get("harmonics", 2))
        if self.H < 1:
            raise ValueError("gait.harmonics must be at least 1")
        self._hk = np.arange(1, self.H + 1, dtype=float)[:, None]   # harmonic numbers
        self.amp2_scale = float(cfg.get("amp2_scale", 1.0))
        self.opt_flags = dict(freq=True, duty=True, amp=True, phase=True, offset=True)
        self.opt_flags.update({k: bool(v) for k, v in cfg.get("optimize", {}).items()})

        self.legs, self.depth = limb_layout(model)
        self.n_types = int(self.depth.max()) + 1 if self.n else 0
        # "family" picks the leg timing. The older "preset_phase" still works the way
        # it always did: only when phase is frozen.
        fam = cfg.get("family")
        if fam is None and not self.opt_flags["phase"]:
            fam = cfg.get("preset_phase")
        self.family = family_key(fam)
        if self.family is not None:
            missing = [self.names[i] for i, L in enumerate(self.legs) if L is None]
            if missing:
                raise ValueError(f"gait family {self.family!r} needs every actuator on a "
                                 f"front/back, left/right leg; cannot place {missing}")
            self._shift = np.array([FAMILIES[self.family][L] for L in self.legs])
        else:
            self._shift = None
        self.preset = self.family or ""          # kept for older callers and the panel

        self._build_index()
        self.base_x = np.full(self.dim, 0.5)
        self._build_mask()

    # ---- parameter indexing: decides which actuators share amp/offset ----
    def _gid_of(self, name):
        for k, g in enumerate(self.groups):
            if g["match"].lower() in name.lower():
                return k
        return None

    def _shared(self, name, field):
        """True if this actuator shares `field` with the rest of its group."""
        gk = self._gid_of(name)
        if gk is None:
            return False
        return field in self.groups[gk].get("share", [])

    def _slot_key(self, i, name, field):
        return ("g", self._gid_of(name)) if self._shared(name, field) else ("i", i)

    def _build_index(self):
        self.amp_idx, self.off_idx = {}, {}
        for i, nm in enumerate(self.names):
            for field, table in (("amp", self.amp_idx), ("offset", self.off_idx)):
                key = self._slot_key(i, nm, field)
                if key not in table:
                    table[key] = len(table)
        self.n_amp = len(self.amp_idx)
        self.n_off = len(self.off_idx)
        # free: a phase per actuator; family: a phase per joint type, shared by legs
        self.n_phase = self.n if self.family is None else self.n_types
        # freq + duty + H * (amp block + phase block) + offset block
        self.dim = 2 + self.H * (self.n_amp + self.n_phase) + self.n_off
        # reverse lookup: actuator i -> its amp slot / offset slot
        self._amp_slot = np.array([self.amp_idx[self._slot_key(i, nm, "amp")]
                                   for i, nm in enumerate(self.names)], dtype=int)
        self._off_slot = np.array([self.off_idx[self._slot_key(i, nm, "offset")]
                                   for i, nm in enumerate(self.names)], dtype=int)

    # ---- parameter blocks and freezing ----
    def blocks(self):
        """Return {block name: (start, stop)} so blocks can be frozen or displayed."""
        b, k = {}, 0
        b["freq"] = (0, 1); k = 1
        b["duty"] = (k, k + 1); k += 1
        for h in range(self.H):
            b[f"amp{h+1}"] = (k, k + self.n_amp); k += self.n_amp
            b[f"phase{h+1}"] = (k, k + self.n_phase); k += self.n_phase
        b["offset"] = (k, k + self.n_off)
        return b

    def _build_mask(self):
        b = self.blocks()
        mask = np.zeros(self.dim, dtype=bool)
        for name, (i0, i1) in b.items():
            if name in ("freq", "duty"):
                key = name
            elif name.startswith("amp"):
                key = "amp"
            elif name.startswith("phase"):
                key = "phase"
            else:
                key = "offset"
            mask[i0:i1] = self.opt_flags.get(key, True)
        self.mask = mask
        self.dim_opt = int(mask.sum())
        # With phase frozen in a family, each joint type keeps a textbook stroke: hip
        # and flipper in phase, knee 60 degrees ahead, no second-harmonic offset.
        if self.family is not None and not self.opt_flags.get("phase", True):
            for name, (i0, i1) in b.items():
                if name == "phase1":
                    frozen = np.zeros(self.n_types)
                    if self.n_types > 1:
                        frozen[1] = 60.0 / 360.0
                    self.base_x[i0:i1] = frozen
                elif name.startswith("phase"):
                    self.base_x[i0:i1] = 0.0

    def expand(self, x_red):
        """Grow a reduced vector (optimized parameters only) into a full one."""
        x_red = np.asarray(x_red, float)
        if x_red.size == self.dim:
            return np.clip(x_red, 0, 1)
        if x_red.size != self.dim_opt:
            raise ValueError(f"expected {self.dim_opt} or {self.dim} parameters, "
                             f"got {x_red.size}")
        full = self.base_x.copy()
        full[self.mask] = np.clip(x_red, 0, 1)
        return full

    def x0_opt(self):
        return self.base_x[self.mask].copy()

    def opt_summary(self):
        on = [k for k, v in self.opt_flags.items() if v]
        off = [k for k, v in self.opt_flags.items() if not v]
        s = f"[search space] optimizing {self.dim_opt}/{self.dim} dims: {'+'.join(on)}"
        if off:
            s += f" | frozen: {'+'.join(off)}"
        if self.family is not None:
            s += (f" | family '{self.family}' ({COMMON_NAMES[self.family]}): leg timing "
                  f"fixed, joint-type phases {'searched' if self.opt_flags['phase'] else 'fixed'}")
        return s

    # ---- optimizer interface: normalized [0,1]^dim <-> physical parameters ----
    def bounds(self):
        return np.zeros(self.dim), np.ones(self.dim)

    def x0(self):
        return np.full(self.dim, 0.5)

    def decode(self, x):
        """Map a normalized vector to physical gait parameters.

        Accepts either a full vector or a reduced one; reduced vectors are expanded
        first, so callers never have to track which of the two they are holding.
        PH is each actuator's phase as seen at a common clock, so tables and plots
        read the same in both modes.
        """
        x = self.expand(x)
        k = 0
        f = self.freq_range[0] + x[k] * (self.freq_range[1] - self.freq_range[0]); k += 1
        duty = self.duty_range[0] + x[k] * (self.duty_range[1] - self.duty_range[0]); k += 1
        amps, blocks = [], []
        for h in range(self.H):
            scale = 1.0 if h == 0 else self.amp2_scale
            amps.append(self.amp_range[0]
                        + x[k:k + self.n_amp] * (self.amp_range[1] - self.amp_range[0]) * scale)
            k += self.n_amp
            blocks.append(x[k:k + self.n_phase] * 2 * np.pi)
            k += self.n_phase
        offs = self.off_range[0] + x[k:k + self.n_off] * (self.off_range[1] - self.off_range[0])
        if self._shift is None:
            u_ph = np.array(blocks)                                   # (H, n)
            phases = list(u_ph)
        else:
            u_ph = np.array([blk[self.depth] for blk in blocks])     # (H, n)
            # the same leg delayed by `shift` looks, at a common clock, like a phase
            # of -h * 2pi * shift on harmonic h
            phases = [(u_ph[h] - (h + 1) * 2 * np.pi * self._shift) % (2 * np.pi)
                      for h in range(self.H)]
        p = dict(freq=float(f), duty=float(duty), A=amps, PH=phases, offs=offs,
                 family=self.family)
        # Per-actuator arrays for ctrl(), gathered once here instead of every step.
        # Treat the returned dict as read-only: ctrl() uses these, not A/PH/offs.
        p["_u_off"] = offs[self._off_slot]
        p["_u_amp"] = np.array([a[self._amp_slot] for a in amps])     # (H, n)
        p["_u_ph"] = u_ph
        p["_u_shift"] = self._shift
        return p

    # ---- called every simulation step ----
    def ctrl(self, x_decoded, t):
        p = x_decoded
        ramp = min(1.0, t / self.ramp_t) if self.ramp_t > 0 else 1.0
        duty = p.get("duty", 0.5)
        shift = p.get("_u_shift")
        if shift is None:
            cycle = (p["freq"] * t) % 1.0                  # progress through one period
            # Phase warp: the first `duty` of the period covers 0..pi (power stroke),
            # the remainder covers pi..2*pi (recovery stroke).
            th = (cycle / duty) * np.pi if cycle < duty \
                else np.pi + ((cycle - duty) / (1.0 - duty)) * np.pi
        else:
            # Each leg runs the same warped stroke, delayed by its family shift.
            cycle = (p["freq"] * t - shift) % 1.0
            th = np.where(cycle < duty, cycle / duty * np.pi,
                          np.pi + (cycle - duty) / (1.0 - duty) * np.pi)
        s = np.sin(self._hk * th + p["_u_ph"])                 # (H, n)
        return p["_u_off"] + ramp * (p["_u_amp"] * s).sum(0)

    def period(self, x_decoded):
        """Seconds per stroke. The duty warp reshapes a stroke but not its length."""
        return 1.0 / x_decoded["freq"]

    # ---- describing a gait ----
    def structure(self, x):
        """What kind of gait is this? Leg timing, and the nearest textbook family.

        Leg timing is read from the joint type that moves most (largest first-harmonic
        amplitude): each leg's delay behind the front-left leg, as a fraction of the
        stroke. The nearest family is the one with the smallest mean circular distance
        from those delays. Returns None if the legs cannot be identified.
        """
        if any(L is None for L in self.legs) or set(self.legs) < set(LEGS):
            return None
        p = self.decode(x)
        amp1 = p["_u_amp"][0]
        types = range(self.n_types)
        type_amp = [float(amp1[self.depth == t].mean()) if np.any(self.depth == t) else 0.0
                    for t in types]
        dom = int(np.argmax(type_amp))
        phase = {}
        for L in LEGS:
            i = next(i for i in range(self.n) if self.legs[i] == L and self.depth[i] == dom)
            phase[L] = p["PH"][0][i]
        delay = {L: float((phase["FL"] - phase[L]) / (2 * np.pi) % 1.0) for L in LEGS}
        dist = {}
        for fam, ref in FAMILIES.items():
            dist[fam] = float(np.mean([abs(_circ(delay[L] - ref[L])) for L in LEGS[1:]]))
        near = min(dist, key=dist.get)
        return dict(delay=delay, dominant_joint=dom, type_amp_deg=np.degrees(type_amp),
                    nearest=near, nearest_common=COMMON_NAMES[near],
                    deviation=dist[near], distances=dist)

    def describe(self, x):
        p = self.decode(x)
        mode = (f"family {self.family} ({COMMON_NAMES[self.family]})"
                if self.family else "free phases")
        lines = [f"freq={p['freq']:.3f} Hz  power-stroke duty={p['duty']*100:.0f}%  "
                 f"(harmonics H={self.H}, {mode})"]
        for i, nm in enumerate(self.names):
            harm = "  ".join(
                f"h{h+1}: amp={p['A'][h][self._amp_slot[i]]:+.3f} "
                f"ph={np.degrees(p['PH'][h][i]):6.1f}deg"
                for h in range(self.H))
            lines.append(f"  {nm:<16} off={p['offs'][self._off_slot[i]]:+.3f}  {harm}")
        st = self.structure(x)
        if st is not None:
            d = st["delay"]
            lines.append("  leg delay behind FL (fraction of a stroke): "
                         + "  ".join(f"{L} {d[L]:.2f}" for L in LEGS))
            lines.append(f"  closest textbook gait: {st['nearest']} "
                         f"({st['nearest_common']}), off by {st['deviation']*100:.0f}% "
                         f"of a stroke on average")
        return "\n".join(lines)

    def info(self):
        placed = sum(L is not None for L in self.legs)
        return (f"[gait] actuators={self.n} on {len(set(L for L in self.legs if L))} legs "
                f"({placed} placed), joint types {self.n_types}, harmonics H={self.H} -> "
                f"dimension={self.dim} (freq 1 + duty 1 + H x (amp {self.n_amp} + phase "
                f"{self.n_phase}) + offset {self.n_off})")

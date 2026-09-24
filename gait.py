# -*- coding: utf-8 -*-
"""
Layer 3, control -- robot-agnostic gait parameterization.

Scans every actuator in the model and drives it with a truncated Fourier series:

    q_i(t) = offset_i + sum_{h=1..H} amp_i^h * sin(2*pi*h*f*t + phase_i^h)

H = 2 is not optional. With a single harmonic the thrust of an antiphase leg pair is
exactly equal and opposite, all four legs cancel, and net thrust is identically zero.
The second harmonic is invariant under a phase shift of pi, which is the mathematical
counterpart of the real robot's passive flipper feathering on the recovery stroke.

Parameter vector = [freq, duty] + per-actuator (amp^h, phase^h) + offsets.
Actuators can share amp/offset through `groups`, which lowers the search dimension:

    groups = [{"match": "1.1", "share": ["amp", "offset"]},
              {"match": "2.1", "share": ["amp", "offset"]}]

Swapping robots changes the actuator count, the parameter vector resizes itself, and
no other module needs to change.
"""
import numpy as np
import mujoco


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
        self.amp2_scale = float(cfg.get("amp2_scale", 1.0))
        self._build_index()
        # Parameter freezing: optimize a subset, hold the rest at base_x.
        self.base_x = np.full(self.dim, 0.5)
        self.opt_flags = dict(freq=True, duty=True, amp=True, phase=True, offset=True)
        self.opt_flags.update({k: bool(v) for k, v in cfg.get("optimize", {}).items()})
        self.preset = cfg.get("preset_phase", "")   # gait used when phase is frozen
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
        self.n_phase = self.n                # phase is always per-actuator
        # freq + duty + H * (amp block + phase block) + offset block
        self.dim = 2 + self.H * (self.n_amp + self.n_phase) + self.n_off
        # reverse lookup: actuator i -> its amp slot / offset slot
        self._amp_slot = [self.amp_idx[self._slot_key(i, nm, "amp")]
                          for i, nm in enumerate(self.names)]
        self._off_slot = [self.off_idx[self._slot_key(i, nm, "offset")]
                          for i, nm in enumerate(self.names)]

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

    def preset_phases(self, name):
        """Phases of a textbook gait, normalized to [0, 1]. name: diag/fb/lr/wave/inphase."""
        two_pi = 2 * np.pi
        table = {
            "diag":    {"FL": 0, "BR": 0, "FR": np.pi, "BL": np.pi},
            "fb":      {"FL": 0, "FR": 0, "BL": np.pi, "BR": np.pi},
            "lr":      {"FL": 0, "BL": 0, "FR": np.pi, "BR": np.pi},
            "wave":    {"FL": 0, "FR": np.pi / 2, "BR": np.pi, "BL": 3 * np.pi / 2},
            "inphase": {"FL": 0, "FR": 0, "BL": 0, "BR": 0},
        }
        t = table.get(name, table["diag"])
        out = np.zeros(self.n_phase)
        for i, nm in enumerate(self.names):
            leg = next((L for L in ("FL", "FR", "BL", "BR") if L in nm), "FL")
            # keep a default intra-leg offset: hip 0 deg, knee 60 deg, wrist 0 deg
            is_knee = (".2" in nm or "2." in nm)
            ph = t[leg] + (np.radians(60) if is_knee else 0.0)
            out[i] = (ph % two_pi) / two_pi
        return out

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
        # Fill base_x with the preset gait, but only when phase is actually frozen.
        # The control panel always writes a preset_phase into its run config; applying
        # it while phase is being optimized would silently move the CMA-ES start point,
        # so the panel and the command line would disagree on identical settings.
        if self.preset and not self.opt_flags.get("phase", True):
            for name, (i0, i1) in b.items():
                if name == "phase1":
                    self.base_x[i0:i1] = self.preset_phases(self.preset)
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
            if self.preset and not self.opt_flags.get("phase", True):
                s += f" (phase pinned to the '{self.preset}' gait)"
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
        """
        x = self.expand(x)
        k = 0
        f = self.freq_range[0] + x[k] * (self.freq_range[1] - self.freq_range[0]); k += 1
        duty = self.duty_range[0] + x[k] * (self.duty_range[1] - self.duty_range[0]); k += 1
        amps, phases = [], []
        for h in range(self.H):
            scale = 1.0 if h == 0 else self.amp2_scale
            amps.append(self.amp_range[0]
                        + x[k:k + self.n_amp] * (self.amp_range[1] - self.amp_range[0]) * scale)
            k += self.n_amp
            phases.append(x[k:k + self.n_phase] * 2 * np.pi)
            k += self.n_phase
        offs = self.off_range[0] + x[k:k + self.n_off] * (self.off_range[1] - self.off_range[0])
        return dict(freq=float(f), duty=float(duty), A=amps, PH=phases, offs=offs)

    def describe(self, x):
        p = self.decode(x)
        lines = [f"freq={p['freq']:.3f} Hz  power-stroke duty={p['duty']*100:.0f}%  "
                 f"(harmonics H={self.H})"]
        for i, nm in enumerate(self.names):
            harm = "  ".join(
                f"h{h+1}: amp={p['A'][h][self._amp_slot[i]]:+.3f} "
                f"ph={np.degrees(p['PH'][h][i]):6.1f}deg"
                for h in range(self.H))
            lines.append(f"  {nm:<16} off={p['offs'][self._off_slot[i]]:+.3f}  {harm}")
        return "\n".join(lines)

    # ---- called every simulation step ----
    def ctrl(self, x_decoded, t):
        p = x_decoded
        ramp = min(1.0, t / self.ramp_t) if self.ramp_t > 0 else 1.0
        duty = p.get("duty", 0.5)
        cycle = (p["freq"] * t) % 1.0                  # progress through one period
        # Phase warp: the first `duty` of the period covers 0..pi (power stroke),
        # the remainder covers pi..2*pi (recovery stroke).
        th = (cycle / duty) * np.pi if cycle < duty \
            else np.pi + ((cycle - duty) / (1.0 - duty)) * np.pi
        u = np.empty(self.n)
        for i in range(self.n):
            val = p["offs"][self._off_slot[i]]
            for h in range(self.H):
                val += ramp * p["A"][h][self._amp_slot[i]] * np.sin((h + 1) * th + p["PH"][h][i])
            u[i] = val
        return u

    def info(self):
        return (f"[gait] actuators={self.n}, harmonics H={self.H} -> dimension={self.dim} "
                f"(freq 1 + duty 1 + H x (amp {self.n_amp} + phase {self.n_phase}) "
                f"+ offset {self.n_off})")

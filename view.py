# -*- coding: utf-8 -*-
"""
Playback -- open a MuJoCo viewer and watch the robot swim.

Usage:
    python view.py config.json                      # play results/best.json
    python view.py config.json results/best.json    # play a specific result
    python view.py config.json --demo               # play a hand-built gait

Viewer: left-drag orbits, right-drag pans, scroll zooms, space pauses, Esc quits.
"""
import sys
import json
import time

import numpy as np
import mujoco
import mujoco.viewer

from simulate import Swimmer, load_cfg

LOOP_SECONDS = 60.0      # restart the playback after this much simulated time


def demo_params(gait):
    """A hand-built diagonal gait with flipper feathering, used before any search."""
    g = gait
    two_pi = 2 * np.pi
    b = g.blocks()
    x = np.full(g.dim, 0.5)

    x[b["freq"][0]] = (1.2 - g.freq_range[0]) / (g.freq_range[1] - g.freq_range[0])
    if "duty" in b:
        x[b["duty"][0]] = 0.5

    k = b["amp1"][0]
    # first-harmonic amplitudes, one per group: hip, knee, flipper
    amps1 = [0.6, 0.3, 0.15][:g.n_amp] + [0.4] * max(0, g.n_amp - 3)
    for j in range(g.n_amp):
        x[k + j] = amps1[j] / g.amp_range[1]
    k += g.n_amp

    # first-harmonic phases: diagonal gait, legs identified by name
    knee_lead = np.radians(60)
    for i, nm in enumerate(g.names):
        leg = next((L for L in ("FL", "FR", "BL", "BR") if L in nm), "FL")
        base = 0.0 if leg in ("FL", "BR") else np.pi
        is_knee = "2." in nm
        ph = base + (knee_lead if is_knee else 0.0)
        x[k + i] = (ph % two_pi) / two_pi
    k += g.n_phase

    if g.H > 1:
        # second harmonic on the flipper only: this is what produces net thrust
        amps2 = [0.0, 0.0, 0.55][:g.n_amp] + [0.0] * max(0, g.n_amp - 3)
        for j in range(g.n_amp):
            x[k + j] = amps2[j] / g.amp_range[1]
        k += g.n_amp
        for i in range(g.n_phase):
            x[k + i] = (np.radians(225) % two_pi) / two_pi
        k += g.n_phase

    x[k:k + g.n_off] = 0.5
    return np.clip(x, 0, 1)


def load_gait(sw, argv):
    """Return (parameter vector, label) from the command line."""
    if "--demo" in argv:
        return demo_params(sw.gait), "hand-built demo gait"
    path = next((a for a in argv[1:] if a.endswith(".json")), "results/best.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return np.array(json.load(fh)["x"], dtype=float), f"optimized gait ({path})"
    except (OSError, KeyError, ValueError) as e:
        print(f"[!] could not read {path} ({e}); falling back to the demo gait")
        return demo_params(sw.gait), "hand-built demo gait"


def main():
    argv = sys.argv[1:]
    cfg_path = argv[0] if argv and argv[0].endswith(".json") else "config.json"
    cfg = load_cfg(cfg_path)
    sw = Swimmer(cfg["model"], cfg)
    print(sw.info())

    x, tag = load_gait(sw, argv)
    try:
        # A saved result may hold only the subset that was optimized, so expand it.
        x = sw.gait.expand(x)
    except ValueError as e:
        print(f"[X] {e}")
        print("    The saved result was produced with a different set of frozen "
              "parameters than the config you just passed.")
        print("    Re-run the optimization, or pass the config that produced it.")
        sys.exit(1)

    print(f"\nplaying: {tag}")
    print(sw.gait.describe(x))
    r = sw.rollout(x)
    print(f"\nthis gait: speed={r['speed']:+.4f} m/s  net displacement={r['dist']:+.3f} m  "
          f"yaw={r['yaw']:.1f} deg\n")

    m, d = sw.model, sw.data
    p = sw.gait.decode(x)
    mujoco.mj_resetData(m, d)
    dt = m.opt.timestep
    settle = int(sw.settle / dt)
    with mujoco.viewer.launch_passive(m, d) as v:
        t0 = time.time()
        k = 0
        while v.is_running():
            t = max(0.0, k * dt - sw.settle)
            d.ctrl[:] = 0 if k < settle else sw.gait.ctrl(p, t)
            sw.hydro.apply(d)
            mujoco.mj_step(m, d)
            if k % 10 == 0:
                v.sync()
                lag = k * dt - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
            k += 1
            if k * dt > LOOP_SECONDS:
                mujoco.mj_resetData(m, d)
                k = 0
                t0 = time.time()


if __name__ == "__main__":
    main()

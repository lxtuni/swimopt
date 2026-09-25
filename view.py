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

from simulate import Swimmer, load_cfg, swimmer_for_result

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

    # first-harmonic phases: a trot, legs and knees identified from the geometry
    knee_lead = np.radians(60)
    if g.family is None:
        for i in range(g.n):
            leg = g.legs[i] or "FL"
            base = 0.0 if leg in ("FL", "BR") else np.pi
            ph = base + (knee_lead if g.depth[i] == 1 else 0.0)
            x[k + i] = (ph % two_pi) / two_pi
    else:
        # in a family the leg timing is fixed; set only each joint type's phase
        for jt in range(g.n_phase):
            x[k + jt] = (knee_lead if jt == 1 else 0.0) / two_pi
    k += g.n_phase

    if g.H > 1:
        # second harmonic on the flipper only, to feather it
        amps2 = [0.0, 0.0, 0.55][:g.n_amp] + [0.0] * max(0, g.n_amp - 3)
        for j in range(g.n_amp):
            x[k + j] = amps2[j] / g.amp_range[1]
        k += g.n_amp
        for i in range(g.n_phase):
            x[k + i] = (np.radians(225) % two_pi) / two_pi
        k += g.n_phase

    x[k:k + g.n_off] = 0.5
    return np.clip(x, 0, 1)


def load_result(cfg, path):
    """(swimmer, full parameter vector, label) for a saved result.

    The swimmer is built with the result's own gait family, so the numbers decode to
    the gait that was found rather than to whatever the config says now.
    """
    with open(path, encoding="utf-8") as fh:
        best = json.load(fh)
    sw = swimmer_for_result(cfg, best)
    return sw, sw.gait.expand(np.array(best["x"], dtype=float)), f"optimized gait ({path})"


def load_gait(sw, argv):
    """Return (parameter vector, label) from the command line, for `sw` as it is.

    Kept for callers that already hold a swimmer; prefer load_result, which also
    picks the right gait family.
    """
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
    path = next((a for a in argv[1:] if a.endswith(".json")), "results/best.json")
    if "--demo" in argv:
        sw = Swimmer(cfg["model"], cfg)
        x, tag = sw.gait.expand(demo_params(sw.gait)), "hand-built demo gait"
    else:
        try:
            sw, x, tag = load_result(cfg, path)
        except (OSError, KeyError, ValueError) as e:
            print(f"[!] could not use {path} ({e}); playing the demo gait instead")
            sw = Swimmer(cfg["model"], cfg)
            x, tag = sw.gait.expand(demo_params(sw.gait)), "hand-built demo gait"
    print(sw.info())

    print(f"\nplaying: {tag}")
    print(sw.gait.describe(x))
    r = sw.rollout(x)
    print(f"\nthis gait: speed={r['speed']:+.4f} m/s  net displacement={r['dist']:+.3f} m  "
          f"yaw={r['yaw']:.1f} deg  peak joint speed {r['peak_joint_speed']:.0f} deg/s\n")

    # Playback goes through Swimmer.rollout, so it runs the same servo model, ramp and
    # physics as the scoring does: what you watch is what was scored.
    dt = sw.model.opt.timestep
    with mujoco.viewer.launch_passive(sw.model, sw.data) as v:
        while v.is_running():
            t0 = time.time()

            def on_step(k, phase):
                if k % 10:
                    return True
                v.sync()
                if not v.is_running():
                    return False
                lag = k * dt - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
                return True

            if sw.rollout(x, on_step=on_step, duration=LOOP_SECONDS) is None:
                break


if __name__ == "__main__":
    main()

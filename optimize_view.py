# -*- coding: utf-8 -*-
"""
Optimization with live rendering -- same CMA-ES search as optimize.py, but every
candidate (or a subset of them) is played back in a MuJoCo viewer as it is evaluated.

Usage:
    python optimize_view.py config.json 200             # show every candidate, 4x speed
    python optimize_view.py config.json 200 --every 5   # show one in five
    python optimize_view.py config.json 200 --best      # show only new records
    python optimize_view.py config.json 200 --rt        # real-time playback

Viewer: drag with the left button to orbit, scroll to zoom. Closing the window ends
the run early and keeps the best result found so far.
"""
import sys
import os
import json
import time
import csv

import numpy as np
import mujoco
import mujoco.viewer
import cma

from simulate import Swimmer, load_cfg


def rollout_view(sw, x, viewer=None, realtime=False, render_every=8):
    """One rollout, optionally rendered. Returns the same keys as Swimmer.rollout.

    Returns None if the viewer window was closed mid-rollout.
    """
    m, d = sw.model, sw.data
    mujoco.mj_resetData(m, d)
    p = sw.gait.decode(x)          # decode expands reduced vectors on its own
    dt = m.opt.timestep
    n_settle, n_steps = int(sw.settle / dt), int(sw.T / dt)
    d.ctrl[:] = 0
    t_wall = time.time()

    for k in range(n_settle):
        sw.hydro.apply(d)
        mujoco.mj_step(m, d)
        if viewer is not None and k % render_every == 0:
            viewer.sync()
            if not viewer.is_running():
                return None

    p0 = d.xpos[sw.trunk_id].copy()
    yaw0 = sw._yaw()
    energy, blew = 0.0, False
    for k in range(n_steps):
        t = k * dt
        d.ctrl[:] = sw.gait.ctrl(p, t)
        sw.hydro.apply(d)
        mujoco.mj_step(m, d)
        if (not np.all(np.isfinite(d.qvel))
                or np.max(np.abs(d.qvel)) > sw.vmax_guard * 20
                or np.linalg.norm(d.cvel[sw.trunk_id, 3:6]) > sw.vmax_guard):
            blew = True
            break
        energy += float(np.sum(np.abs(d.actuator_force * d.actuator_velocity))) * dt
        if viewer is not None and k % render_every == 0:
            viewer.sync()
            if not viewer.is_running():
                return None
            if realtime:
                lag = (n_settle + k) * dt - (time.time() - t_wall)
                if lag > 0:
                    time.sleep(lag)

    if blew:
        return dict(ok=False, fitness=-1e3, dist=0.0, speed=0.0, yaw=0.0,
                    energy=0.0, power=0.0)

    disp = d.xpos[sw.trunk_id].copy() - p0
    yaw_drift = abs(np.degrees(sw._wrap(sw._yaw() - yaw0)))
    dist = float(disp[:2] @ sw._forward_dir(yaw0))
    speed = dist / sw.T
    power = energy / sw.T
    fit = speed - sw.w_yaw * np.radians(yaw_drift) / sw.T - sw.w_energy * power
    return dict(ok=True, fitness=float(fit), dist=dist, speed=speed,
                yaw=float(yaw_drift), energy=float(energy), power=float(power))


def parse_args(argv):
    cfg_path = argv[0] if argv and argv[0].endswith(".json") else "config.json"
    budget = next((int(a) for a in argv if a.isdigit()), None)
    every = int(argv[argv.index("--every") + 1]) if "--every" in argv else 1
    return cfg_path, budget, every, "--best" in argv, "--rt" in argv


def main():
    cfg_path, budget, every, best_only, realtime = parse_args(sys.argv[1:])

    cfg = load_cfg(cfg_path)
    sw = Swimmer(cfg["model"], cfg)
    print(sw.info())
    budget = budget or cfg.get("budget", 250)
    outdir = cfg.get("outdir", "results")
    os.makedirs(outdir, exist_ok=True)

    es = cma.CMAEvolutionStrategy(
        sw.gait.x0_opt(), cfg.get("sigma0", 0.25),
        {"bounds": [0, 1], "popsize": cfg.get("popsize", 10),
         "maxfevals": budget, "verbose": -9, "seed": cfg.get("seed", 1)})

    log = open(os.path.join(outdir, "log.csv"), "w", newline="", encoding="utf-8")
    wr = csv.writer(log)
    wr.writerow(["eval", "gen", "fitness", "speed", "yaw", "ok"]
                + [f"x{i}" for i in range(sw.gait.dim_opt)])

    n_eval, gen, t0 = 0, 0, time.time()
    best = dict(fitness=-1e9, speed=0.0, yaw=0.0, x=list(map(float, sw.gait.x0_opt())))
    mode = ("new records only" if best_only
            else (f"one in {every}" if every > 1 else "every candidate"))
    print(f"\nstarting search (budget {budget}, showing {mode}, "
          f"{'real-time' if realtime else 'fast-forward'} playback)")
    print("closing the window ends the run early and keeps the best result\n")

    with mujoco.viewer.launch_passive(sw.model, sw.data) as v:
        stop = False
        while not es.stop() and n_eval < budget and not stop:
            X = es.ask()
            F = []
            for x in X:
                n_eval += 1
                show = v.is_running() and (not best_only) and (n_eval % every == 0)
                r = rollout_view(sw, x, v if show else None, realtime)
                if r is None:                     # window closed
                    stop = True
                    break
                F.append(-r["fitness"])
                wr.writerow([n_eval, gen, f"{r['fitness']:.5f}", f"{r['speed']:.5f}",
                             f"{r['yaw']:.2f}", int(r["ok"])] + [f"{q:.4f}" for q in x])
                flag = ""
                if r["fitness"] > best["fitness"]:
                    best = dict(fitness=r["fitness"], speed=r["speed"], yaw=r["yaw"],
                                x=list(map(float, x)))
                    with open(os.path.join(outdir, "best.json"), "w", encoding="utf-8") as fh:
                        json.dump(best, fh, indent=1)
                    flag = "  * NEW BEST"
                    if best_only and v.is_running():     # replay it immediately
                        rollout_view(sw, x, v, realtime)
                print(f"  #{n_eval:4d}  speed {r['speed']:+.4f} m/s  "
                      f"yaw {r['yaw']:5.1f} deg  fitness {r['fitness']:+.4f}{flag}")
            if F and not stop:
                es.tell(X[:len(F)], F)
                gen += 1
                print(f"--- gen {gen:3d} | evals {n_eval:4d} | best {best['fitness']:+.4f} "
                      f"(speed {best['speed']:.4f} m/s) | sigma {es.sigma:.4f} "
                      f"| {time.time()-t0:.0f}s ---")
    log.close()

    print("\n=== best gait ===")
    print(sw.gait.describe(best["x"]))
    print(f"speed {best['speed']:.4f} m/s | yaw {best['yaw']:.1f} deg | "
          f"{n_eval} rollouts | {time.time()-t0:.0f}s")
    print(f"saved to {outdir}/best.json -- run 3_view_best.bat to replay it")


if __name__ == "__main__":
    main()

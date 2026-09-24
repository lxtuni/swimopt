# -*- coding: utf-8 -*-
"""
Layer 4, optimization -- CMA-ES gait search (Hansen & Ostermeier 2001).

Robot-agnostic: it only ever calls Swimmer.rollout(x).

Usage:
    python optimize.py config.json [evaluations]

Writes results/log.csv (one row per evaluation), results/best.json (best parameters)
and results/convergence.json (best-so-far history).
"""
import sys
import os
import json
import time
import csv

import cma

from simulate import Swimmer, load_cfg


def main(cfg_path, budget=None):
    cfg = load_cfg(cfg_path)
    sw = Swimmer(cfg["model"], cfg)
    print(sw.info())
    dim = sw.gait.dim_opt
    budget = int(budget or cfg.get("budget", 500))
    outdir = cfg.get("outdir", "results")
    os.makedirs(outdir, exist_ok=True)

    es = cma.CMAEvolutionStrategy(
        sw.gait.x0_opt(), cfg.get("sigma0", 0.25),
        {"bounds": [0, 1], "popsize": cfg.get("popsize", 10),
         "maxfevals": budget, "verbose": -9, "seed": cfg.get("seed", 1)})

    log = open(os.path.join(outdir, "log.csv"), "w", newline="", encoding="utf-8")
    wr = csv.writer(log)
    wr.writerow(["eval", "gen", "fitness", "speed", "yaw", "ok"] + [f"x{i}" for i in range(dim)])

    n_eval, gen, t0 = 0, 0, time.time()
    best = dict(fitness=-1e9, speed=0.0, yaw=0.0, x=list(map(float, sw.gait.x0_opt())))
    hist = []
    while not es.stop() and n_eval < budget:
        X = es.ask()
        F = []
        for x in X:
            r = sw.rollout(x)
            F.append(-r["fitness"])
            n_eval += 1
            wr.writerow([n_eval, gen, f"{r['fitness']:.5f}", f"{r['speed']:.5f}",
                         f"{r['yaw']:.2f}", int(r["ok"])] + [f"{v:.4f}" for v in x])
            if r["fitness"] > best["fitness"]:
                best = dict(fitness=r["fitness"], speed=r["speed"], yaw=r["yaw"],
                            x=list(map(float, x)))
                with open(os.path.join(outdir, "best.json"), "w", encoding="utf-8") as fh:
                    json.dump(best, fh, indent=1)
        es.tell(X, F)
        gen += 1
        hist.append(best["fitness"])
        print(f"--- gen {gen:3d} | evals {n_eval:4d} | best {best['fitness']:+.4f} "
              f"(speed {best['speed']:.4f} m/s, yaw {best['yaw']:.1f} deg) "
              f"| sigma {es.sigma:.4f} | {time.time()-t0:.0f}s ---")
    log.close()
    with open(os.path.join(outdir, "convergence.json"), "w", encoding="utf-8") as fh:
        json.dump({"history": hist}, fh)

    print("\n=== best gait ===")
    print(sw.gait.describe(best["x"]))
    print(f"speed {best['speed']:.4f} m/s | yaw {best['yaw']:.1f} deg | "
          f"{n_eval} rollouts | {time.time()-t0:.0f}s")
    print(f"saved to {outdir}/best.json, log.csv, convergence.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.json",
         sys.argv[2] if len(sys.argv) > 2 else None)

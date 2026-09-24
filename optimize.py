# -*- coding: utf-8 -*-
"""
Layer 4, optimization -- CMA-ES gait search (Hansen & Ostermeier 2001).

Robot-agnostic: it only ever calls Swimmer.rollout(x). Candidates within a generation
are independent, so they are evaluated in parallel worker processes; every rollout is
deterministic, so a parallel search returns exactly what a serial one would.

Usage:
    python optimize.py config.json [evaluations]

Writes to results/ (or cfg["outdir"]):
    run_config.json   the exact configuration and library versions used
    log.csv           one row per evaluation
    best.json         the best gait, as a full parameter vector
    convergence.json  best-so-far fitness and sigma per generation
"""
import sys
import os
import csv
import time
import multiprocessing as mp

import numpy as np
import mujoco

from simulate import Swimmer, load_cfg, record_best, atomic_json_dump

LOG_FIELDS = ["eval", "gen", "fitness", "speed", "heading_rms", "roll_rms", "pitch_rms",
              "power", "thrust_lift", "thrust_drag", "feasible", "ok"]

# ---------- worker processes ----------
_WORKER = None          # this process's own Swimmer; MuJoCo models are not shareable


def _init_worker(cfg):
    global _WORKER
    _WORKER = Swimmer(cfg["model"], cfg)


def _eval_worker(x):
    r = _WORKER.rollout(x)
    r.pop("traj", None)
    return r


def n_workers(cfg):
    """cfg["workers"]: an integer, or "auto" for one per candidate up to the CPU count."""
    w = cfg.get("workers", "auto")
    if w in (None, "auto"):
        w = min(int(cfg.get("popsize", 10)), os.cpu_count() or 1)
    return max(1, int(w))


def open_pool(cfg):
    """A worker pool for cfg, or None when one worker is all there is."""
    w = n_workers(cfg)
    if w <= 1:
        return None
    return mp.get_context("spawn").Pool(w, initializer=_init_worker, initargs=(cfg,))


def cma_options(cfg, budget):
    seed = cfg.get("seed", 1)
    if not seed:
        print("[!] seed 0 (or empty) makes cma choose a time-based seed, so this run "
              "will NOT be reproducible. Use any non-zero integer.")
    return {"bounds": [0, 1], "popsize": cfg.get("popsize", 10),
            "maxfevals": budget, "verbose": -9, "seed": seed}


def _candidate_line(n, r, flag):
    # The control panel parses this line; keep the "#<n>" and "fitness" fields.
    over = "" if r["feasible"] or not r["ok"] else "  over limit"
    return (f"  #{n:4d}  speed {r['speed']:+.4f} m/s  RMS head {r['heading_rms']:4.1f} "
            f"roll {r['roll_rms']:4.1f} pitch {r['pitch_rms']:4.1f} deg  "
            f"fitness {r['fitness']:+.4f}{over}{flag}")


def search(sw, cfg, budget, pool=None, should_show=None, render=None, on_record=None):
    """Run one CMA-ES search. Returns (best, n_eval, finished).

    pool        worker pool from open_pool(); None evaluates in this process.
    should_show f(eval_index) -> bool. Candidates it selects are evaluated here,
                through render(x), which may return None to stop the search (the
                live viewer's window was closed).
    on_record   f(x), called on every new best (the viewer's records-only mode).
    """
    import cma                      # not needed by worker processes
    outdir = cfg.get("outdir", "results")
    os.makedirs(outdir, exist_ok=True)
    atomic_json_dump({"config": cfg, "mujoco": mujoco.__version__,
                      "cma": cma.__version__, "numpy": np.__version__,
                      "budget": budget}, os.path.join(outdir, "run_config.json"),
                     indent=1, ensure_ascii=False)

    es = cma.CMAEvolutionStrategy(sw.gait.x0_opt(), cfg.get("sigma0", 0.25),
                                  cma_options(cfg, budget))
    log = open(os.path.join(outdir, "log.csv"), "w", newline="", encoding="utf-8")
    wr = csv.writer(log)
    wr.writerow(LOG_FIELDS + [f"x{i}" for i in range(sw.gait.dim_opt)])

    n_eval, gen, t0 = 0, 0, time.time()
    best = record_best(sw.diverged(), sw.gait.x0_opt(), sw.gait)
    best["fitness"] = -1e9
    hist, sigmas = [], []
    finished = True
    try:
        while not es.stop() and n_eval < budget:
            X = es.ask()
            ids = [n_eval + i + 1 for i in range(len(X))]
            shown = {i for i in range(len(X)) if should_show and should_show(ids[i])}
            hidden = [i for i in range(len(X)) if i not in shown]
            results = [None] * len(X)
            if hidden:
                xs = [X[i] for i in hidden]
                rs = pool.map(_eval_worker, xs) if pool else [sw.rollout(x) for x in xs]
                for i, r in zip(hidden, rs):
                    results[i] = r
            for i in sorted(shown):
                results[i] = render(X[i])
                if results[i] is None:
                    finished = False
                    break

            F = []
            for i, x in enumerate(X):
                r = results[i]
                if r is None:
                    break
                n_eval += 1
                wr.writerow([n_eval, gen, f"{r['fitness']:.5f}", f"{r['speed']:.5f}",
                             f"{r['heading_rms']:.2f}", f"{r['roll_rms']:.2f}",
                             f"{r['pitch_rms']:.2f}", f"{r['power']:.3f}",
                             f"{r['thrust_lift']:.5f}", f"{r['thrust_drag']:.5f}",
                             int(r["feasible"]), int(r["ok"])] + [f"{v:.6f}" for v in x])
                flag = ""
                if r["fitness"] > best["fitness"]:
                    best = record_best(r, x, sw.gait)
                    atomic_json_dump(best, os.path.join(outdir, "best.json"), indent=1)
                    flag = "  * NEW BEST"
                print(_candidate_line(n_eval, r, flag))
                if flag and on_record is not None:
                    on_record(x)
                F.append(-r["fitness"])
            if not finished or len(F) < len(X):
                finished = False
                break
            es.tell(X, F)
            gen += 1
            hist.append(best["fitness"])
            sigmas.append(float(es.sigma))
            print(f"--- gen {gen:3d} | evals {n_eval:4d} | best {best['fitness']:+.4f} "
                  f"(speed {best['speed']:.4f} m/s, "
                  f"{'within limits' if best['feasible'] else 'over limit'}) "
                  f"| sigma {es.sigma:.4f} | {time.time()-t0:.0f}s ---", flush=True)
    finally:
        log.close()
        atomic_json_dump({"history": hist, "sigma": sigmas},
                         os.path.join(outdir, "convergence.json"))
    best["elapsed_s"] = time.time() - t0
    return best, n_eval, finished


def print_summary(sw, best, n_eval, outdir):
    print("\n=== best gait ===")
    print(sw.gait.describe(best["x"]))
    print(f"speed {best['speed']:.4f} m/s ({best['bl_s']:.2f} BL/s) | RMS heading "
          f"{best['heading_rms']:.1f}, roll {best['roll_rms']:.1f}, pitch "
          f"{best['pitch_rms']:.1f} deg | {best['power']:.1f} W | "
          f"{'within limits' if best['feasible'] else 'OVER LIMIT'} | "
          f"{n_eval} rollouts | {best.get('elapsed_s', 0):.0f}s")
    print(f"saved to {outdir}/best.json, log.csv, convergence.json, run_config.json")


def main(cfg_path, budget=None):
    cfg = load_cfg(cfg_path)
    sw = Swimmer(cfg["model"], cfg)
    print(sw.info())
    budget = int(budget or cfg.get("budget", 500))
    pool = open_pool(cfg)
    print(f"[search] budget {budget}, "
          f"{n_workers(cfg) if pool else 1} worker process(es)", flush=True)
    try:
        best, n_eval, _ = search(sw, cfg, budget, pool=pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print_summary(sw, best, n_eval, cfg.get("outdir", "results"))
    return best


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.json",
         sys.argv[2] if len(sys.argv) > 2 else None)

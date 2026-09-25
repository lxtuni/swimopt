# -*- coding: utf-8 -*-
"""
Run the same search with several seeds and keep every result.

    python tools/multi_seed.py [evaluations] [--seeds N] [--config config.json]
                               [--out results_seeds]

A single CMA-ES run on this problem is a lower bound: identical settings with
different seeds have landed 0.17 to 0.34 m/s apart. This runs N seeds, writes each to
<out>/s<seed>/, copies the best one to results/best.json so the viewer and the panel
show it, and prints the spread. Finished seeds are skipped on a rerun.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                       # noqa: E402

from simulate import load_cfg            # noqa: E402
import optimize                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _arg(args, name, default):
    return args[args.index(name) + 1] if name in args else default


def main():
    os.chdir(ROOT)
    args = sys.argv[1:]
    budget = next((int(a) for a in args if a.isdigit()), None)
    seeds = int(_arg(args, "--seeds", 3))
    cfg_path = _arg(args, "--config", "config.json")
    out = _arg(args, "--out", "results_seeds")
    base = load_cfg(cfg_path)
    budget = budget or int(base.get("budget", 4000))

    bests = {}
    for s in range(1, seeds + 1):
        d = os.path.join(out, f"s{s}")
        done = os.path.join(d, "done.json")
        if os.path.exists(done):
            bests[s] = json.load(open(done, encoding="utf-8"))
            print(f"seed {s}: already done ({bests[s]['speed']:.4f} m/s)")
            continue
        c = dict(base, outdir=d, seed=s)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, "_cfg.json")
        json.dump(c, open(path, "w", encoding="utf-8"), indent=1)
        print(f"\n=== seed {s} of {seeds}, {budget} evaluations ===", flush=True)
        bests[s] = optimize.main(path, budget)
        json.dump(bests[s], open(done, "w", encoding="utf-8"), indent=1)

    feasible = {s: b for s, b in bests.items() if b["feasible"]}
    pool = feasible or bests
    top = max(pool, key=lambda s: pool[s]["fitness"])
    os.makedirs("results", exist_ok=True)
    shutil.copy(os.path.join(out, f"s{top}", "best.json"), os.path.join("results", "best.json"))
    v = np.array([b["speed"] for b in bests.values()])
    print(f"\nspeeds by seed: " + ", ".join(f"s{s} {b['speed']:.4f}" for s, b in bests.items()))
    print(f"mean {v.mean():.4f} m/s, sd {v.std(ddof=1) if v.size > 1 else 0:.4f}, "
          f"{len(feasible)}/{len(bests)} within limits")
    print(f"best: seed {top}, copied to results/best.json")


if __name__ == "__main__":
    main()

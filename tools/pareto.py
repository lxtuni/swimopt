# -*- coding: utf-8 -*-
"""
Speed against power: the least power each physics needs to swim at a given speed.

    python tools/pareto.py [evaluations] [--seeds N] [--speeds 0.04,0.08,0.12,0.16]

For every target speed v, and for resistive-only and lift-augmented physics, it
minimizes mean power subject to: speed >= v, and the straight-and-level limits. That
traces the lower edge of the speed-power trade-off for each physics, so the two can
be compared at matched speed -- the comparison the single-optimum study could not
make, because the lift-based winners there were also slower, and power rises with
speed.

Finished runs are kept in results_pareto/ and skipped on a rerun. Writes a table, a
summary.json and docs/pareto.png.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                       # noqa: E402

from simulate import Swimmer, load_cfg   # noqa: E402
import optimize                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_pareto")
PHYSICS = (("drag", False), ("lift", True))
# Reference categorical palette, slots 1-2 (validated adjacent and all-pairs).
COLOR = {"drag": "#2a78d6", "lift": "#eb6834"}
LABEL = {"drag": "resistive only", "lift": "resistive + lift"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _arg(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def run(base, phys, lift, v, seed, budget):
    d = os.path.join(OUT, f"{phys}_v{v:.3f}_s{seed}")
    done = os.path.join(d, "done.json")
    if os.path.exists(done):
        b = json.load(open(done, encoding="utf-8"))
        if b.get("_budget") == budget:
            print(f"  {phys} v>={v:.3f} seed {seed}: already done")
            return b
    c = copy.deepcopy(base)
    c["hydro"]["lift"] = lift
    c["objective"] = {"mode": "power", "min_speed": v}
    c.update(outdir=d, seed=seed)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "_cfg.json")
    json.dump(c, open(path, "w", encoding="utf-8"), indent=1)
    print(f"\n=== {phys}  speed >= {v:.3f} m/s  seed {seed}  budget {budget} ===", flush=True)
    b = optimize.main(path, budget)
    b["_budget"] = budget
    json.dump(b, open(done, "w", encoding="utf-8"), indent=1)
    return b


def plot(front, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for phys, pts in front.items():
        if not pts["best"]:
            continue
        allx = [q[0] for q in pts["all"]]
        ally = [q[1] for q in pts["all"]]
        ax.scatter(allx, ally, s=22, color=COLOR[phys], alpha=0.35, linewidths=0)
        bx = [q[0] for q in pts["best"]]
        by = [q[1] for q in pts["best"]]
        ax.plot(bx, by, "-o", color=COLOR[phys], linewidth=2, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABEL[phys])
        ax.annotate(LABEL[phys], (bx[-1], by[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("speed (m/s)", fontsize=9, color=INK2)
    ax.set_ylabel("least mean power (W)", fontsize=9, color=INK2)
    ax.set_title("Least power needed to swim straight and level at each speed",
                 loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.savefig(out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    os.chdir(ROOT)
    args = sys.argv[1:]
    budget = next((int(a) for a in args if a.isdigit()), 2500)
    seeds = int(_arg(args, "--seeds", 2))
    speeds = [float(v) for v in _arg(args, "--speeds", "0.04,0.08,0.12,0.16").split(",")]
    base = load_cfg(os.path.join(ROOT, "config.json"))

    won = {(p, v, s): run(base, p, lift, v, s, budget)
           for p, lift in PHYSICS for v in speeds for s in range(1, seeds + 1)}

    print(f"\n\n{'='*92}\n  LEAST POWER AT EACH REQUIRED SPEED  ({seeds} seeds, {budget} "
          f"evaluations each; best seed shown)\n{'='*92}")
    print(f"{'speed >=':>10}" + "".join(f"{LABEL[p]:>28}" for p, _ in PHYSICS) + f"{'ratio':>12}")
    front = {p: {"best": [], "all": []} for p, _ in PHYSICS}
    rows = []
    for v in speeds:
        cells = {}
        for p, _ in PHYSICS:
            feas = [won[(p, v, s)] for s in range(1, seeds + 1) if won[(p, v, s)]["feasible"]]
            for b in feas:
                front[p]["all"].append((b["speed"], b["power"]))
            best = min(feas, key=lambda b: b["power"]) if feas else None
            cells[p] = best
            if best:
                front[p]["best"].append((best["speed"], best["power"]))
        txt = []
        for p, _ in PHYSICS:
            b = cells[p]
            txt.append(f"{b['power']:7.2f} W @ {b['speed']:.3f} m/s" if b else "no feasible gait")
        ratio = (cells["lift"]["power"] / cells["drag"]["power"]
                 if cells["drag"] and cells["lift"] else float("nan"))
        print(f"{v:>10.3f}" + "".join(f"{t:>28}" for t in txt) + f"{ratio:>12.2f}")
        rows.append({"min_speed": v, **{p: cells[p] and {k: cells[p][k] for k in
                     ("speed", "power", "heading_rms", "roll_rms", "pitch_rms",
                      "thrust_lift", "thrust_drag", "peak_joint_speed")} for p, _ in PHYSICS}})
    json.dump({"budget": budget, "seeds": seeds, "rows": rows},
              open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=1)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    plot(front, os.path.join(ROOT, "docs", "pareto.png"))
    print("\nratio = lift power / drag power at the same required speed; below 1 means "
          "lift is cheaper")
    print("written results_pareto/summary.json and docs/pareto.png")


if __name__ == "__main__":
    main()

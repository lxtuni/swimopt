# -*- coding: utf-8 -*-
"""
Speed against power: the least power each physics needs to swim at a given speed.

    python tools/pareto.py [evaluations] [--seeds N] [--speeds 0.10,0.20,0.30]
                           [--families diag,lr,fb,wave,inphase] [--surfacing 0.05]

For every target speed v, for resistive-only and for lift-augmented physics, and for
every gait family, it minimizes mean power subject to: speed >= v, the straight-and-
level limits, and a limb out of the water at most --surfacing of the time (the model
has no free-surface physics, so a gait rowing across the surface is outside what it
predicts). The lower envelope over families and seeds is that physics' speed-power
front. Comparing the two fronts at matched speed is the fair drag-versus-lift question:
the single-optimum study could not answer it, because power rises steeply with speed.

Families rather than free phases: the family study showed the 35-dim free search
converges poorly within a practical budget (free winners were 3-5x slower than the
family winners they resembled), while a family search is 17-dim and reliable.

Finished runs are kept in results_pareto/ and skipped on a rerun. Writes a table,
results_pareto/summary.json and docs/pareto.png.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                       # noqa: E402

from simulate import Swimmer, load_cfg   # noqa: E402
from gait import COMMON_NAMES            # noqa: E402
import optimize                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_pareto")
PHYSICS = (("drag", False), ("lift", True))
FAMILIES = "diag,lr,fb,wave,inphase"
# Reference categorical palette, slots 1-2 (validated adjacent and all-pairs).
COLOR = {"drag": "#2a78d6", "lift": "#eb6834"}
MARK = {"drag": "o", "lift": "s"}        # shape as well as colour
LABEL = {"drag": "resistive only", "lift": "resistive + lift"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
KEEP = ("speed", "bl_s", "power", "heading_rms", "roll_rms", "pitch_rms", "thrust_lift",
        "thrust_drag", "lift_share", "peak_joint_speed", "torque_sat", "surfacing", "feasible")


def _arg(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def study_cfg(base, lift, family, v, surfacing):
    c = copy.deepcopy(base)
    c["hydro"]["lift"] = lift
    c["gait"]["family"] = family
    c["gait"].pop("preset_phase", None)
    c["objective"] = {"mode": "power", "min_speed": v}
    c["limits"] = dict(c.get("limits") or {}, surfacing=surfacing)
    return c


def run(base, phys, lift, family, v, seed, budget, surfacing):
    d = os.path.join(OUT, f"{phys}_{family}_v{v:.2f}_s{seed}")
    done = os.path.join(d, "done.json")
    if os.path.exists(done):
        b = json.load(open(done, encoding="utf-8"))
        if b.get("_budget") == budget and b.get("_surfacing") == surfacing:
            print(f"  {phys} {family} v>={v:.2f} seed {seed}: already done")
            return b
    c = study_cfg(base, lift, family, v, surfacing)
    c.update(outdir=d, seed=seed)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "_cfg.json")
    json.dump(c, open(path, "w", encoding="utf-8"), indent=1)
    print(f"\n=== {phys}  {family}  speed >= {v:.2f} m/s  seed {seed}  budget {budget} ===",
          flush=True)
    b = optimize.main(path, budget)
    b.update(_budget=budget, _surfacing=surfacing)
    json.dump(b, open(done, "w", encoding="utf-8"), indent=1)
    return b


def replay(base, phys, lift, family, v, surfacing, best, cache):
    """Re-run a winner to get every metric, the lift/drag thrust split included."""
    key = (phys, family, v)
    if key not in cache:
        c = study_cfg(base, lift, family, v, surfacing)
        cache[key] = Swimmer(c["model"], c)
    r = cache[key].rollout(np.array(best["x"]))
    out = {k: (bool(r[k]) if k == "feasible" else float(r[k])) for k in KEEP}
    out.update(family=family, min_speed=v)
    return out


def plot(pts, env, out):
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
    for phys, _ in PHYSICS:
        feas = [r for r in pts[phys] if r["feasible"]]
        ax.scatter([r["speed"] for r in feas], [r["power"] for r in feas], s=18,
                   marker=MARK[phys], color=COLOR[phys], alpha=0.25, linewidths=0)
        e = [r for r in env[phys] if r]
        if not e:
            continue
        ax.plot([r["speed"] for r in e], [r["power"] for r in e], "-" + MARK[phys],
                color=COLOR[phys], linewidth=2, markersize=6.5, markeredgecolor=SURFACE,
                markeredgewidth=1.5)
        ax.annotate(LABEL[phys], (e[-1]["speed"], e[-1]["power"]), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK2)
        for r in e:                       # which family wins each envelope point
            ax.annotate(COMMON_NAMES[r["family"]], (r["speed"], r["power"]),
                        xytext=(0, 8 if phys == "lift" else -12), textcoords="offset points",
                        ha="center", fontsize=7, color=INK2)
    ax.set_xlabel("speed (m/s)", fontsize=9, color=INK2)
    ax.set_ylabel("least mean power (W)", fontsize=9, color=INK2)
    ax.set_title("Least power to swim straight, level and submerged at each speed\n"
                 "line: best over all gait families; faint: every family's optimum",
                 loc="left", fontsize=10, color=INK)
    ax.margins(x=0.12)
    fig.savefig(out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    os.chdir(ROOT)
    args = sys.argv[1:]
    budget = next((int(a) for a in args if a.isdigit()), 2000)
    seeds = int(_arg(args, "--seeds", 1))
    speeds = [float(v) for v in _arg(args, "--speeds", "0.10,0.20,0.30").split(",")]
    fams = _arg(args, "--families", FAMILIES).split(",")
    surfacing = float(_arg(args, "--surfacing", 0.05))
    base = load_cfg(os.path.join(ROOT, "config.json"))

    # cheapest targets first, so a partial study already yields whole low-speed rows
    won = {(p, f, v, s): run(base, p, lift, f, v, s, budget, surfacing)
           for v in speeds for p, lift in PHYSICS for f in fams for s in range(1, seeds + 1)}

    cache = {}
    pts = {p: [] for p, _ in PHYSICS}
    env = {p: [] for p, _ in PHYSICS}
    for v in speeds:
        for p, lift in PHYSICS:
            rs = [replay(base, p, lift, f, v, surfacing, won[(p, f, v, s)], cache)
                  for f in fams for s in range(1, seeds + 1)]
            pts[p] += rs
            feas = [r for r in rs if r["feasible"]]
            env[p].append(min(feas, key=lambda r: r["power"]) if feas else None)

    print(f"\n\n{'='*100}\n  LEAST POWER AT EACH REQUIRED SPEED  (best of {len(fams)} families x "
          f"{seeds} seed(s), {budget} evaluations each, surfacing <= {surfacing:.0%})\n{'='*100}")
    print(f"{'speed >=':>9}" + "".join(f"{LABEL[p]:>36}" for p, _ in PHYSICS)
          + f"{'lift/drag':>11}")
    for i, v in enumerate(speeds):
        cells = []
        for p, _ in PHYSICS:
            r = env[p][i]
            cells.append(f"{r['power']:6.2f} W  {COMMON_NAMES[r['family']]:<5} "
                         f"lift imp {r['thrust_lift']:+.2f} N*s" if r else "no feasible gait")
        d, l = env["drag"][i], env["lift"][i]
        ratio = l["power"] / d["power"] if d and l else float("nan")
        print(f"{v:>9.2f}" + "".join(f"{c:>36}" for c in cells) + f"{ratio:>11.2f}")
    print("\nper family (W; '-' = no feasible gait):")
    print(f"{'':>14}" + "".join(f"{COMMON_NAMES[f]:>8}" for f in fams))
    for p, _ in PHYSICS:
        for v in speeds:
            row = []
            for f in fams:
                ok = [r for r in pts[p] if r["family"] == f and r["min_speed"] == v
                      and r["feasible"]]
                row.append(f"{min(r['power'] for r in ok):8.2f}" if ok else f"{'-':>8}")
            print(f"{p:>5} >= {v:.2f}" + "".join(row))
    print("\nlift imp = forward impulse of the lift force over the scored window "
          "(negative = lift opposes the motion); resistive-only physics has none")
    print("lift/drag = least power with lift / without, at the same required speed; "
          "below 1 means the lift model makes swimming cheaper")

    json.dump({"budget": budget, "seeds": seeds, "families": fams, "surfacing": surfacing,
               "speeds": speeds, "envelope": env, "points": pts},
              open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8"), indent=1)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    plot(pts, env, os.path.join(ROOT, "docs", "pareto.png"))
    print("written results_pareto/summary.json and docs/pareto.png")


if __name__ == "__main__":
    main()

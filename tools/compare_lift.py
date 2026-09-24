# -*- coding: utf-8 -*-
"""
Resistive versus lift-augmented propulsion: a 2x2 experiment over several seeds.

    python tools/compare_lift.py [evaluations] [--seeds N] [--popsize P]

Factors:
  physics    hydro.lift off ("drag") or on ("lift")
  objective  the config's straight-and-level limits ("limits"), or none ("free")

The limits are part of the design, not a detail. Lift on a flipper necessarily puts a
moment on the hull, so constraining attitude constrains lift indirectly; measuring at
one setting alone cannot separate "lift is a worse way to swim" from "lift is being
taxed for tilting the robot".

Each cell runs N independent seeds, so a difference between cells can be compared with
the spread inside a cell. Every winner is then re-scored under both physics models,
which separates "lift changes which gait is best" from "lift changes what every gait
scores". Finished runs are kept in results_cmp/<cell>_s<seed>/ and skipped on a rerun,
so an interrupted study resumes where it stopped.

Reported quantities are physical: m/s, body lengths/s, degrees, watts, and the cost of
transport in J/m. Fitness is never compared across objectives.
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
OUT = os.path.join(ROOT, "results_cmp")
NO_LIMITS = {"heading_deg": 180, "roll_deg": 180, "pitch_deg": 180}
CELLS = [(lift, lim) for lim in (True, False) for lift in (False, True)]


def tag(lift, lim):
    return f"{'lift' if lift else 'drag'}_{'limits' if lim else 'free'}"


def cell_cfg(base, lift, lim):
    c = copy.deepcopy(base)
    c["hydro"]["lift"] = lift
    if not lim:
        c["limits"] = dict(NO_LIMITS)
    return c


def run(base, lift, lim, seed, budget, popsize):
    out = os.path.join(OUT, f"{tag(lift, lim)}_s{seed}")
    best_path = os.path.join(out, "best.json")
    if os.path.exists(best_path):
        with open(best_path, encoding="utf-8") as fh:
            best = json.load(fh)
        if best.get("_budget") == budget:
            print(f"  {tag(lift, lim)} seed {seed}: already done, skipping")
            return best
    c = cell_cfg(base, lift, lim)
    c.update(outdir=out, seed=seed, popsize=popsize)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "_cfg.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(c, fh, indent=1)
    print(f"\n=== {tag(lift, lim)}  seed {seed}  budget {budget}  popsize {popsize} ===",
          flush=True)
    best = optimize.main(path, budget)
    best["_budget"] = budget
    with open(best_path, "w", encoding="utf-8") as fh:
        json.dump(best, fh, indent=1)
    return best


def evaluate(base, lift, lim, x):
    c = cell_cfg(base, lift, lim)
    sw = Swimmer(c["model"], c)
    x = np.asarray(x, float)
    return sw.rollout(x), sw.gait.decode(x)


def stat(vals):
    a = np.array(vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "n/a"
    sd = a.std(ddof=1) if a.size > 1 else 0.0
    return f"{a.mean():.3f} ± {sd:.3f}"


def straight_and_level(r, limits):
    """Judged against the config's limits, whatever objective the gait came from."""
    return (r["heading_rms"] <= limits["heading_deg"] and r["roll_rms"] <= limits["roll_deg"]
            and r["pitch_rms"] <= limits["pitch_deg"])


def main():
    os.chdir(ROOT)
    args = sys.argv[1:]
    budget = next((int(a) for a in args if a.isdigit()), 4000)
    seeds = int(args[args.index("--seeds") + 1]) if "--seeds" in args else 3
    popsize = int(args[args.index("--popsize") + 1]) if "--popsize" in args else 16
    base = load_cfg(os.path.join(ROOT, "config.json"))
    from simulate import DEFAULT_LIMITS
    limits = dict(DEFAULT_LIMITS, **base.get("limits", {}))

    won = {}
    for lift, lim in CELLS:
        for s in range(1, seeds + 1):
            won[(lift, lim, s)] = run(base, lift, lim, s, budget, popsize)

    rows = {}
    for lift, lim in CELLS:
        rs = []
        for s in range(1, seeds + 1):
            r, p = evaluate(base, lift, lim, won[(lift, lim, s)]["x"])
            r["freq"] = p["freq"]
            # cost of transport: energy per metre; undefined for a gait going nowhere
            r["cot"] = r["power"] / r["speed"] if r["speed"] > 1e-3 else float("nan")
            r["level"] = straight_and_level(r, limits)
            rs.append(r)
        rows[(lift, lim)] = rs

    W = 20
    print(f"\n\n{'='*113}\n  EACH CELL: mean ± sd over {seeds} seeds, {budget} evaluations each"
          f"\n{'='*113}")
    print(f"{'cell':<13}{'speed m/s':>{W}}{'BL/s':>{W}}{'freq Hz':>{W}}{'power W':>{W}}"
          f"{'COT J/m':>{W}}")
    for key, rs in rows.items():
        print(f"{tag(*key):<13}{stat([r['speed'] for r in rs]):>{W}}"
              f"{stat([r['bl_s'] for r in rs]):>{W}}{stat([r['freq'] for r in rs]):>{W}}"
              f"{stat([r['power'] for r in rs]):>{W}}{stat([r['cot'] for r in rs]):>{W}}")
    print(f"\n{'cell':<13}{'RMS heading':>{W}}{'RMS roll':>{W}}{'RMS pitch':>{W}}"
          f"{'fwd lift N*s':>{W}}{'straight+level':>{W}}")
    for key, rs in rows.items():
        print(f"{tag(*key):<13}{stat([r['heading_rms'] for r in rs]):>{W}}"
              f"{stat([r['roll_rms'] for r in rs]):>{W}}"
              f"{stat([r['pitch_rms'] for r in rs]):>{W}}"
              f"{stat([r['thrust_lift'] for r in rs]):>{W}}"
              f"{sum(r['level'] for r in rs):>{W-3}}/{len(rs)}")
    print(f"(straight+level is judged against the config limits for every cell: "
          f"{limits['heading_deg']:g}/{limits['roll_deg']:g}/{limits['pitch_deg']:g} deg)")

    print(f"\n{'='*96}\n  CROSS-EVALUATION: each winner re-scored under the OTHER physics"
          f"\n  (same objective as its cell; speed in m/s)\n{'='*96}")
    cross = {}
    for lim in (True, False):
        for lift in (False, True):
            own, other = [], []
            for s in range(1, seeds + 1):
                x = won[(lift, lim, s)]["x"]
                own.append(evaluate(base, lift, lim, x)[0])
                other.append(evaluate(base, not lift, lim, x)[0])
            cross[(lift, lim)] = (own, other)
            print(f"{tag(lift, lim):<13} own physics {stat([r['speed'] for r in own]):>16}"
                  f"   other physics {stat([r['speed'] for r in other]):>16}"
                  f"   straight+level there: "
                  f"{sum(straight_and_level(r, limits) for r in other)}/{seeds}")

    summary = {tag(*k): [{kk: r[kk] for kk in ("speed", "bl_s", "freq", "power", "cot",
                                                "heading_rms", "roll_rms", "pitch_rms",
                                                "thrust_lift", "thrust_drag", "level")}
                         for r in rs] for k, rs in rows.items()}
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"budget": budget, "seeds": seeds, "popsize": popsize,
                   "cells": summary}, fh, indent=1, default=float)
    print(f"\nwritten {os.path.relpath(os.path.join(OUT, 'summary.json'), ROOT)}")
    print("Caveats: cl is the flat-plate textbook value, not measured on the real "
          "flipper; freq_range caps the stroke rate at the servo limit, so a winner on "
          "that bound was chosen by the bound.")


if __name__ == "__main__":
    main()

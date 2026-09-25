# -*- coding: utf-8 -*-
"""
Compare gait families, each at its own best stroke.

    python tools/gait_families.py [evaluations] [--seeds N] [--lift]
                                  [--families free,diag,lr,fb,wave,inphase]
                                  [--limits HEADING,ROLL,PITCH]

--limits overrides the RMS attitude limits in degrees, e.g. --limits 10,5,5 for a
stricter level; results then go to results_families_lim10-5-5/ (plus _lift).

For each family the legs keep that family's timing -- trot, pace, bound, walk, pronk --
and the search tunes everything else: frequency, duty cycle, amplitudes, offsets, and
the phase of each joint type (hip, knee, flipper) shared by all legs. "free" lets every
actuator's phase vary independently, so it can find coordinations no family names; its
winner is then classified by the nearest family.

This is the fair version of "which gait is best": a family is not judged by one
hand-picked stroke but by the best stroke it admits. Every family gets the same budget
and the same seeds. Finished runs are kept in results_families/ and skipped on a rerun.
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
DEFAULT = ["free", "diag", "lr", "fb", "wave", "inphase"]


def _arg(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def family_cfg(base, family, lift, limits=None):
    c = copy.deepcopy(base)
    if limits:
        c["limits"] = dict(c.get("limits") or {}, heading_deg=limits[0], roll_deg=limits[1],
                           pitch_deg=limits[2])
    c["hydro"]["lift"] = lift
    c["gait"]["family"] = None if family == "free" else family
    c["gait"].pop("preset_phase", None)
    return c


def run(base, family, seed, budget, lift, out, limits=None):
    d = os.path.join(out, f"{family}_s{seed}")
    done = os.path.join(d, "done.json")
    if os.path.exists(done):
        b = json.load(open(done, encoding="utf-8"))
        if b.get("_budget") == budget:
            print(f"  {family} seed {seed}: already done")
            return b
    c = family_cfg(base, family, lift, limits)
    c.update(outdir=d, seed=seed)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "_cfg.json")
    json.dump(c, open(path, "w", encoding="utf-8"), indent=1)
    print(f"\n=== {family}  seed {seed}  budget {budget} ===", flush=True)
    b = optimize.main(path, budget)
    b["_budget"] = budget
    json.dump(b, open(done, "w", encoding="utf-8"), indent=1)
    return b


def stat(v):
    a = np.array(v, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return "n/a"
    return f"{a.mean():.3f} ± {a.std(ddof=1) if a.size > 1 else 0:.3f}"


def main():
    os.chdir(ROOT)
    args = sys.argv[1:]
    budget = next((int(a) for a in args if a.isdigit()), 2500)
    seeds = int(_arg(args, "--seeds", 3))
    lift = "--lift" in args
    fams = _arg(args, "--families", ",".join(DEFAULT)).split(",")
    limits = [float(v) for v in _arg(args, "--limits").split(",")] if "--limits" in args else None
    tag = "_lim" + "-".join(f"{v:g}" for v in limits) if limits else ""
    out = os.path.join(ROOT, "results_families" + tag + ("_lift" if lift else ""))
    base = load_cfg(os.path.join(ROOT, "config.json"))

    won = {(f, s): run(base, f, s, budget, lift, out, limits) for f in fams
           for s in range(1, seeds + 1)}

    rows = {}
    for f in fams:
        c = family_cfg(base, f, lift, limits)
        sw = Swimmer(c["model"], c)
        rs = []
        for s in range(1, seeds + 1):
            x = np.array(won[(f, s)]["x"])
            r = sw.rollout(x)
            p = sw.gait.decode(x)
            st = sw.gait.structure(x)
            r.update(freq=p["freq"], duty=p["duty"],
                     cot=r["power"] / r["speed"] if r["speed"] > 1e-3 else float("nan"),
                     nearest=st["nearest"] if st else None,
                     deviation=st["deviation"] if st else float("nan"))
            rs.append(r)
        rows[f] = rs

    W = 18
    print(f"\n\n{'='*112}\n  GAIT FAMILIES: mean ± sd over {seeds} seeds, {budget} evaluations each"
          f"{', lift on' if lift else ''}"
          f"{', limits heading/roll/pitch %g/%g/%g deg' % tuple(limits) if limits else ''}"
          f"\n{'='*112}")
    print(f"{'family':<18}{'speed m/s':>{W}}{'freq Hz':>{W}}{'duty':>{W}}{'power W':>{W}}"
          f"{'COT J/m':>{W}}{'feasible':>10}")
    for f, rs in rows.items():
        name = f if f == "free" else f"{f} ({COMMON_NAMES[f]})"
        print(f"{name:<18}{stat([r['speed'] for r in rs]):>{W}}{stat([r['freq'] for r in rs]):>{W}}"
              f"{stat([r['duty'] for r in rs]):>{W}}{stat([r['power'] for r in rs]):>{W}}"
              f"{stat([r['cot'] for r in rs]):>{W}}{sum(r['feasible'] for r in rs):>7}/{len(rs)}")
    if "free" in rows:
        print("\nfree-phase winners, classified:")
        for s, r in enumerate(rows["free"], 1):
            print(f"  seed {s}: {r['speed']:.3f} m/s, nearest {r['nearest']} "
                  f"({COMMON_NAMES.get(r['nearest'], '?')}), off by "
                  f"{r['deviation']*100:.0f}% of a stroke")
    summary = {f: [{k: (float(r[k]) if isinstance(r[k], (int, float, np.floating)) else r[k])
                    for k in ("speed", "bl_s", "freq", "duty", "power", "cot", "heading_rms",
                              "roll_rms", "pitch_rms", "peak_joint_speed", "surfacing",
                              "feasible", "nearest", "deviation")} for r in rs]
               for f, rs in rows.items()}
    json.dump({"budget": budget, "seeds": seeds, "lift": lift, "limits": limits,
               "families": summary},
              open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), indent=1)
    print(f"\nwritten {os.path.relpath(os.path.join(out, 'summary.json'), ROOT)}")


if __name__ == "__main__":
    main()

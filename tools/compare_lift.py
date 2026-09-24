# -*- coding: utf-8 -*-
"""
Resistive versus lift-augmented propulsion, as a 2x2 experiment.

    python tools/compare_lift.py [evaluations]

Two factors, because they interact and reporting either alone is misleading:

  physics   hydro.lift off or on
  objective w_attitude 0 or the config value

The attitude penalty is a confound for this question. Lift on a flipper necessarily
produces a moment about the hull, so penalizing attitude penalizes lift indirectly. A
lift-versus-drag comparison run only at one attitude weight cannot separate "lift is a
worse way to swim" from "lift is being taxed for tilting the robot".

Every winner is then scored under both physics models. The cross-evaluation separates
"lift changes which gait is best" from "lift changes what every gait scores".

Results are reported as physical quantities. Fitness values from different weights are
not comparable and are shown only within a fixed objective.

Outputs go to results_<case>/, leaving results/ alone.
"""
import os
import sys
import copy
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                       # noqa: E402

from simulate import Swimmer, load_cfg   # noqa: E402
import optimize                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def case_tag(lift_on, att_on):
    return f"{'lift' if lift_on else 'drag'}_{'att' if att_on else 'noatt'}"


def build_cfg(base, lift_on, att_on):
    cfg = copy.deepcopy(base)
    cfg["hydro"]["lift"] = lift_on
    if not att_on:
        cfg["w_attitude"] = 0.0
    return cfg


def run_case(base, lift_on, att_on, budget):
    """One search. Returns the config used and the best gait found."""
    tag = case_tag(lift_on, att_on)
    cfg = build_cfg(base, lift_on, att_on)
    cfg["outdir"] = f"results_{tag}"
    cfg_path = os.path.join(ROOT, f"_cmp_{tag}.json")
    with open(cfg_path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)

    print(f"\n{'='*78}\n  CASE '{tag}': hydro.lift={lift_on}, "
          f"w_attitude={cfg.get('w_attitude')}, budget {budget}\n{'='*78}")
    optimize.main(cfg_path, budget)
    with open(os.path.join(ROOT, cfg["outdir"], "best.json"), encoding="utf-8") as fh:
        best = json.load(fh)
    os.remove(cfg_path)
    return cfg, best


def evaluate_under(base, lift_on, att_on, x):
    """Score a gait under chosen physics and objective, wherever it came from."""
    cfg = build_cfg(base, lift_on, att_on)
    sw = Swimmer(cfg["model"], cfg)
    return sw.rollout(sw.gait.expand(np.asarray(x, float))), sw


def main():
    os.chdir(ROOT)
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    base = load_cfg(os.path.join(ROOT, "config.json"))
    conditions = [(lift, att) for lift in (False, True) for att in (False, True)]

    won = {}
    for lift_on, att_on in conditions:
        cfg, best = run_case(base, lift_on, att_on, budget)
        won[(lift_on, att_on)] = best

    print(f"\n\n{'='*78}\n  WHAT EACH SEARCH FOUND, under the conditions it searched in"
          f"\n{'='*78}")
    print(f"{'search':<14}{'speed m/s':>11}{'BL/s':>8}{'freq Hz':>9}{'roll':>8}"
          f"{'pitch':>8}{'yaw':>8}{'power W':>10}{'lift share':>12}")
    freqs = {}
    for lift_on, att_on in conditions:
        tag = case_tag(lift_on, att_on)
        r, sw = evaluate_under(base, lift_on, att_on, won[(lift_on, att_on)]["x"])
        p = sw.gait.decode(sw.gait.expand(won[(lift_on, att_on)]["x"]))
        freqs[(lift_on, att_on)] = p["freq"]
        print(f"{tag:<14}{r['speed']:>11.4f}{r['bl_s']:>8.3f}{p['freq']:>9.3f}"
              f"{r['roll_amp']:>8.1f}{r['pitch_amp']:>8.1f}{r['yaw']:>8.1f}"
              f"{r['power']:>10.1f}{r['lift_share']:>12.3f}")

    print(f"\n{'='*78}\n  CROSS-EVALUATION: every winner re-scored under BOTH physics"
          f"\n  (speed is physical and always comparable; fitness only within a column)"
          f"\n{'='*78}")
    print(f"{'gait from':<14}{'scored under':<16}{'speed m/s':>11}{'fwd lift':>10}"
          f"{'fwd drag':>10}{'roll':>8}{'power W':>10}")
    cross = {}
    for lift_on, att_on in conditions:
        for phys in (False, True):
            r, _ = evaluate_under(base, phys, att_on, won[(lift_on, att_on)]["x"])
            cross[(lift_on, att_on, phys)] = r
            print(f"{case_tag(lift_on, att_on):<14}"
                  f"{('lift physics' if phys else 'drag physics'):<16}"
                  f"{r['speed']:>11.4f}{r['thrust_lift']:>10.3f}"
                  f"{r['thrust_drag']:>10.3f}"
                  f"{r['roll_amp']:>8.1f}{r['power']:>10.1f}")
    print("fwd lift / fwd drag: signed impulse along the heading, N*s. "
          "Positive drives the robot forwards.")

    print(f"\n{'='*78}\n  READING\n{'='*78}")
    for att_on in (False, True):
        label = "with attitude penalty" if att_on else "no attitude penalty  "
        d = cross[(False, att_on, True)]["speed"]   # drag-found gait, judged with lift
        l = cross[(True, att_on, True)]["speed"]    # lift-found gait, judged with lift
        faster = "the lift-aware search" if l > d else "the resistive search"
        print(f"{label}: under lift physics the lift-aware winner swims {l:.4f} m/s "
              f"and the resistive winner {d:.4f} m/s -> {faster} is faster.")
    s_noatt = cross[(True, False, True)]["lift_share"]
    s_att = cross[(True, True, True)]["lift_share"]
    print(f"\nLift share of forward impulse in the lift-aware winners: "
          f"{s_noatt:.2f} without the attitude penalty, {s_att:.2f} with it.")
    print(f"Stroke frequency: {freqs[(False, True)]:.2f} Hz resistive, "
          f"{freqs[(True, True)]:.2f} Hz lift-aware (both with attitude penalty).")
    print("\nCaveats that belong with any use of these numbers:")
    print("  - cl = flat-plate textbook value, NOT measured on the real flipper.")
    print("  - one seed, one budget. Re-run with other seeds before trusting a margin.")
    print("  - a frequency sitting on its range bound means the bound, not the "
          "physics, chose it.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Collect every finished optimization run into thesis_notes/data/all_runs.csv.

    python thesis_notes/export_data.py

Reads each results_*/<run>/ folder: the metrics saved with the winner (done.json or
best.json, measured by the scoring rollout itself) and the exact config the run used
(_cfg.json). Stroke frequency, duty cycle and nearest gait family are recomputed from
the saved parameter vector with that config, and the winner is replayed once to add
the electrical power (shaft power plus copper loss), which older runs did not save.
Nothing is re-optimized.
"""
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np                      # noqa: E402
from simulate import Swimmer            # noqa: E402

# which study each results folder belongs to, and whether its numbers still stand
STUDIES = {
    "results_servo": ("free_phase_servo", "valid"),
    "results_families": ("families_speed", "valid"),
    "results_families_lim10-5-5": ("families_speed_strict", "valid"),
    "results_families_lim10-5-5_lift": ("families_speed_strict", "valid"),
    "results_pareto": ("speed_power", "valid"),
    "results_pareto_lim10-5-5": ("speed_power_strict", "valid"),
    "results_pareto_ls0.1": ("speed_power_dragstroke", "valid"),
    # run before the servo model existed: joint speeds of 2700-4200 deg/s
    "results_cmp": ("drag_vs_lift_2x2_preservo", "WITHDRAWN (no servo speed limit)"),
}
COLS = ["study", "status", "run", "physics", "family", "seed", "objective", "min_speed",
        "limit_heading", "limit_roll", "limit_pitch", "limit_surfacing",
        "limit_lift_thrust_share", "servo_dps",
        "budget", "feasible", "speed_m_s", "bl_s", "power_W", "cot_J_m", "power_el_W",
        "cot_el_J_m", "freq_Hz", "duty",
        "heading_rms", "roll_rms", "pitch_rms", "impulse_lift_Ns", "impulse_drag_Ns",
        "lift_thrust_share",
        "peak_joint_speed_dps", "torque_sat", "surfacing", "nearest_family", "deviation"]


def rows():
    cache = {}
    for folder, (study, status) in STUDIES.items():
        for d in sorted(glob.glob(os.path.join(folder, "*"))):
            res = next((p for p in (os.path.join(d, "done.json"), os.path.join(d, "best.json"))
                        if os.path.exists(p)), None)
            cfgp = os.path.join(d, "_cfg.json")
            if not res or not os.path.exists(cfgp):
                continue
            b = json.load(open(res, encoding="utf-8"))
            c = json.load(open(cfgp, encoding="utf-8"))
            key = json.dumps({k: c[k] for k in ("model", "gait", "servo") if k in c},
                             sort_keys=True) + str(c["hydro"].get("lift"))
            if key not in cache:
                cache[key] = Swimmer(c["model"], c)
            g = cache[key].gait
            x = np.array(b["x"])
            p = g.decode(x)
            st = g.structure(x)
            if "power_el" in b and "lift_thrust_share" in b:
                pel, lts = float(b["power_el"]), float(b["lift_thrust_share"])
            else:                       # older runs: replay the winner once
                r = cache[key].rollout(x)
                pel, lts = float(r["power_el"]), float(r["lift_thrust_share"])
            lim = c.get("limits") or {}
            obj = c.get("objective") or {}
            spd = float(b["speed"])
            yield {
                "study": study, "status": status, "run": os.path.relpath(d, ROOT),
                "physics": "drag+lift" if c["hydro"].get("lift") else "drag",
                "family": c["gait"].get("family") or "free", "seed": c.get("seed"),
                "objective": obj.get("mode", "speed"),
                "min_speed": obj.get("min_speed") if obj.get("mode") == "power" else "",
                "limit_heading": lim.get("heading_deg"), "limit_roll": lim.get("roll_deg"),
                "limit_pitch": lim.get("pitch_deg"), "limit_surfacing": lim.get("surfacing"),
                "limit_lift_thrust_share": lim.get("lift_thrust_share"),
                "servo_dps": (c.get("servo") or {}).get("max_speed_dps"),
                "budget": b.get("_budget", c.get("budget")), "feasible": b["feasible"],
                "speed_m_s": round(spd, 4), "bl_s": round(b["bl_s"], 3),
                "power_W": round(b["power"], 3),
                "cot_J_m": round(b["power"] / spd, 2) if spd > 1e-3 else "",
                "power_el_W": round(pel, 3) if pel == pel else "",
                "cot_el_J_m": round(pel / spd, 2) if spd > 1e-3 and pel == pel else "",
                "freq_Hz": round(p["freq"], 3), "duty": round(p["duty"], 3),
                "heading_rms": round(b["heading_rms"], 2), "roll_rms": round(b["roll_rms"], 2),
                "pitch_rms": round(b["pitch_rms"], 2),
                "impulse_lift_Ns": round(b.get("thrust_lift", 0.0), 3),
                "impulse_drag_Ns": round(b.get("thrust_drag", 0.0), 3),
                "lift_thrust_share": round(lts, 3),
                "peak_joint_speed_dps": round(b.get("peak_joint_speed", float("nan")), 0),
                "torque_sat": round(b.get("torque_sat", float("nan")), 3),
                "surfacing": round(b.get("surfacing", float("nan")), 3),
                "nearest_family": st["nearest"] if st else "",
                "deviation": round(st["deviation"], 3) if st else "",
            }


def main():
    out = os.path.join(HERE, "data", "all_runs.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rs = list(rows())
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rs)
    print(f"{len(rs)} runs -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()

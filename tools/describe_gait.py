# -*- coding: utf-8 -*-
"""
Describe a gait: what kind it is, and a figure of how it moves.

    python tools/describe_gait.py [results/best.json] [--config config.json]
                                  [--lift] [--out docs/gait_diagram.png]

Prints the gait's parameters, each leg's timing behind the front-left leg, the
nearest textbook gait family (trot, pace, bound, walk, pronk) and the validity checks.
Saves a figure with:

  a gait diagram -- for each leg, when it is in its power stroke, i.e. when the leg
                    tip moves backward relative to the body, over the last strokes;
  joint angles   -- hip, knee and flipper of every leg over the same strokes;
  speed          -- forward speed of the trunk over the scored window;
  attitude       -- heading, roll and pitch against their limits.

The rollout is the scoring rollout itself (Swimmer.rollout with a callback), so the
figure shows exactly the motion that was scored, servo model included.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                       # noqa: E402
import mujoco                            # noqa: E402

from simulate import Swimmer, load_cfg   # noqa: E402
import view as view_mod                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGS = ("FL", "FR", "BL", "BR")
# Reference categorical palette, slots 1-4 in order (validated adjacent-pairs by its
# authors, light mode). Yellow and aqua sit below 3:1 on the surface, so every leg is
# also identified by position or by a direct label, never by colour alone.
LEG_COLOR = {"FL": "#2a78d6", "FR": "#eb6834", "BL": "#1baf7a", "BR": "#eda100"}
LEG_STYLE = {"FL": "-", "FR": "-", "BL": "--", "BR": "--"}   # front solid, back dashed
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
JOINT_NAMES = ("hip", "knee", "flipper", "joint 4", "joint 5")


def _arg(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def record(sw, x):
    """Run the scoring rollout and record what the figure needs, every step."""
    m, d = sw.model, sw.data
    g = sw.gait
    tid = sw.trunk_id
    tips = {}
    for L in LEGS:
        idx = [i for i in range(g.n) if g.legs[i] == L]
        if idx:
            deepest = max(idx, key=lambda i: g.depth[i])
            tips[L] = int(m.jnt_bodyid[m.actuator_trnid[deepest, 0]])
    qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    rec = {k: [] for k in ("t", "q", "speed", "heading", "roll", "pitch")}
    rec["tip_x"] = {L: [] for L in tips}
    state = {}
    dt = m.opt.timestep

    def on_step(k, phase):
        if phase != "run":
            return True
        R = d.xmat[tid].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        if not state:
            state["yaw0"] = yaw
            state["fwd"] = np.array([math.cos(yaw), math.sin(yaw)])
            state["n"] = 0
        rec["t"].append(state["n"] * dt)
        state["n"] += 1
        rec["q"].append(np.degrees(d.qpos[qadr]).copy())
        rec["speed"].append(float(d.qvel[0:2] @ state["fwd"]))
        rec["heading"].append(math.degrees((yaw - state["yaw0"] + math.pi) % (2 * math.pi)
                                           - math.pi))
        rec["roll"].append(math.degrees(math.atan2(R[2, 1], R[2, 2])))
        rec["pitch"].append(math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))))
        for L, b in tips.items():
            rec["tip_x"][L].append(float((R.T @ (d.xipos[b] - d.xpos[tid]))[0]))
        return True

    r = sw.rollout(x, on_step=on_step)
    for k in ("t", "speed", "heading", "roll", "pitch"):
        rec[k] = np.array(rec[k])
    rec["q"] = np.array(rec["q"])
    # power stroke: the leg tip moving backward relative to the body
    rec["power"] = {L: np.gradient(np.array(v), dt) < 0 for L, v in rec["tip_x"].items()}
    return r, rec


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def figure(sw, p, r, rec, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = sw.gait
    per = g.period(p)
    t = rec["t"]
    show = t >= t[-1] - 3 * per                       # the last three strokes
    ts = t[show] - t[show][0]

    fig = plt.figure(figsize=(11, 10), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.4, 1.0, 1.0], hspace=0.55, wspace=0.28)
    fig.suptitle(title, x=0.06, ha="left", fontsize=12, color=INK, fontweight="bold")

    # a) gait diagram
    ax = fig.add_subplot(gs[0, :])
    _style(ax)
    for row, L in enumerate(LEGS):
        on = rec["power"].get(L)
        if on is None:
            continue
        on = on[show]
        edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.broken_barh([(ts[a], ts[min(b, len(ts) - 1)] - ts[a])], (row - 0.32, 0.64),
                           facecolors=LEG_COLOR[L], edgecolor=SURFACE, linewidth=1.0)
    ax.set_yticks(range(len(LEGS)), LEGS)
    ax.set_ylim(len(LEGS) - 0.5, -0.5)
    ax.set_xlim(0, ts[-1])
    ax.set_title("Gait diagram: power stroke (leg tip moving backward relative to the body)",
                 loc="left", fontsize=9.5, color=INK)
    ax.set_xlabel("time within the last three strokes (s)", fontsize=8.5, color=INK2)
    ax.grid(False, axis="y")

    # b) joint angles, one panel per joint type
    for jt in range(min(g.n_types, 3)):
        ax = fig.add_subplot(gs[1, jt])
        _style(ax)
        ends = []
        for L in LEGS:
            idx = [i for i in range(g.n) if g.legs[i] == L and g.depth[i] == jt]
            if not idx:
                continue
            y = rec["q"][show, idx[0]]
            ax.plot(ts, y, LEG_STYLE[L], color=LEG_COLOR[L], linewidth=1.6)
            ends.append([y[-1], L])
        # direct labels at the line ends, nudged apart so none overlap
        lo, hi = ax.get_ylim()
        gap = 0.075 * (hi - lo)
        ends.sort()
        for k in range(1, len(ends)):
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
        for yv, L in ends:
            ax.annotate(L, (ts[-1], yv), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK2, annotation_clip=False)
        ax.set_title(f"{JOINT_NAMES[jt]} angle (deg)", loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("time (s)", fontsize=8.5, color=INK2)
    fig.legend(handles=[matplotlib.lines.Line2D([], [], color=LEG_COLOR[L], linestyle=LEG_STYLE[L],
                                                label=L) for L in LEGS],
               loc="upper right", bbox_to_anchor=(0.98, 0.735), ncol=4, frameon=False,
               fontsize=8.5, labelcolor=INK2)

    # c) forward speed over the whole scored window
    ax = fig.add_subplot(gs[2, :])
    _style(ax)
    ax.plot(t, rec["speed"], color=INK, linewidth=1.2)
    ax.axhline(r["speed"], color=INK2, linewidth=1.0, linestyle=":")
    ax.annotate(f"mean {r['speed']:.3f} m/s", (t[-1], r["speed"]), xytext=(-4, 6),
                textcoords="offset points", ha="right", fontsize=8, color=INK2)
    ax.set_title("Forward speed of the trunk (m/s)", loc="left", fontsize=9.5, color=INK)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("time after the ramp-in (s)", fontsize=8.5, color=INK2)

    # d) attitude: small multiples, each against its limit
    for col, key in enumerate(("heading", "roll", "pitch")):
        ax = fig.add_subplot(gs[3, col])
        _style(ax)
        lim = sw.limits[key]
        ax.axhspan(-lim, lim, color=GRID, alpha=0.5, linewidth=0)
        ax.plot(t, rec[key], color=INK, linewidth=1.0)
        rms = r[f"{key}_rms"]
        ax.set_title(f"{key} (deg), RMS {rms:.1f} / limit {lim:g}", loc="left",
                     fontsize=9.5, color=INK)
        ax.set_xlim(0, t[-1])
        ax.set_xlabel("time (s)", fontsize=8.5, color=INK2)

    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    args = sys.argv[1:]
    os.chdir(ROOT)
    result = next((a for a in args if a.endswith(".json") and not a.startswith("--")
                   and a != _arg(args, "--config")), "results/best.json")
    cfg = load_cfg(_arg(args, "--config", "config.json"))
    if "--lift" in args:
        cfg["hydro"]["lift"] = True
    out = _arg(args, "--out", os.path.join("docs", "gait_diagram.png"))
    sw, x, label = view_mod.load_result(cfg, result)
    p = sw.gait.decode(x)

    print(f"{label}\n")
    print(sw.gait.describe(x))
    r, rec = record(sw, x)
    print(f"\nspeed {r['speed']:.4f} m/s ({r['bl_s']:.2f} BL/s), {r['power']:.2f} W, "
          f"RMS heading {r['heading_rms']:.1f} / roll {r['roll_rms']:.1f} / pitch "
          f"{r['pitch_rms']:.1f} deg, {'within limits' if r['feasible'] else 'OVER LIMIT'}")
    print(f"validity: peak joint speed {r['peak_joint_speed']:.0f} deg/s, torque saturated "
          f"{r['torque_sat']*100:.0f}% of the time, a limb out of the water "
          f"{r['surfacing']*100:.0f}% of the time")
    for L in LEGS:
        on = rec["power"].get(L)
        if on is not None:
            print(f"  {L}: power stroke {on.mean()*100:.0f}% of the time")

    st = sw.gait.structure(x)
    kind = (f"{st['nearest_common']}-like ({st['nearest']}), off by "
            f"{st['deviation']*100:.0f}% of a stroke" if st else "legs not identified")
    title = (f"{p['freq']:.2f} Hz, duty {p['duty']*100:.0f}%, {r['speed']:.3f} m/s, "
             f"{r['power']:.1f} W  |  {kind}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    figure(sw, p, r, rec, out, title)
    print(f"\nfigure written to {out}")


if __name__ == "__main__":
    main()

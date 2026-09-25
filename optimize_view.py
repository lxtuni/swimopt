# -*- coding: utf-8 -*-
"""
Optimization with live rendering -- the same CMA-ES search as optimize.py, with some
or all candidates played back in a MuJoCo viewer as they are evaluated. Candidates
that are not shown still run in parallel worker processes.

Usage:
    python optimize_view.py config.json 200             # show every candidate
    python optimize_view.py config.json 200 --every 5   # show one in five
    python optimize_view.py config.json 200 --best      # show only new records
    python optimize_view.py config.json 200 --rt        # real-time playback

Viewer: drag with the left button to orbit, scroll to zoom. Closing the window ends
the run early and keeps the best result found so far.
"""
import sys
import time

import mujoco
import mujoco.viewer

import optimize
from simulate import Swimmer, load_cfg


def rollout_view(sw, x, viewer=None, realtime=False, render_every=8):
    """One rollout, rendered into `viewer` if given. Same result as Swimmer.rollout.

    Returns None if the viewer window was closed. There is no simulation loop here:
    rendering happens through Swimmer.rollout's on_step callback, so what you watch
    is exactly what gets scored.
    """
    if viewer is None:
        return sw.rollout(x)
    dt = sw.model.opt.timestep
    t_wall = time.time()

    def on_step(k, phase):
        if k % render_every:
            return True
        viewer.sync()
        if not viewer.is_running():
            return False
        if realtime and phase != "settle":
            lag = k * dt - (time.time() - t_wall)
            if lag > 0:
                time.sleep(lag)
        return True

    return sw.rollout(x, on_step=on_step)


def parse_args(argv):
    cfg_path = argv[0] if argv and argv[0].endswith(".json") else "config.json"
    budget = next((int(a) for a in argv if a.isdigit()), None)
    every = int(argv[argv.index("--every") + 1]) if "--every" in argv else 1
    return cfg_path, budget, every, "--best" in argv, "--rt" in argv


def main():
    cfg_path, budget, every, best_only, realtime = parse_args(sys.argv[1:])
    cfg = load_cfg(cfg_path)
    sw = Swimmer(cfg["model"], cfg)
    print(sw.info())
    budget = int(budget or cfg.get("budget", 250))
    mode = ("new records only" if best_only
            else (f"one in {every}" if every > 1 else "every candidate"))
    print(f"\nstarting search (budget {budget}, showing {mode}, "
          f"{'real-time' if realtime else 'fast-forward'} playback)")
    print("closing the window ends the run early and keeps the best result\n", flush=True)

    pool = None if (not best_only and every == 1) else optimize.open_pool(cfg)
    try:
        with mujoco.viewer.launch_passive(sw.model, sw.data) as v:
            def show(i):
                # A closed window routes the next candidate to render(), which stops
                # the search -- otherwise it would carry on invisibly to the end.
                if not v.is_running():
                    return True
                return (not best_only) and i % every == 0

            def render(x):
                if not v.is_running():
                    return None
                return rollout_view(sw, x, v, realtime)

            def replay(x):
                if v.is_running():
                    rollout_view(sw, x, v, realtime)

            best, n_eval, finished = optimize.search(
                sw, cfg, budget, pool=pool, should_show=show, render=render,
                on_record=replay if best_only else None)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    optimize.print_summary(sw, best, n_eval, cfg.get("outdir", "results"))
    if not finished:
        print("(stopped early: the viewer window was closed)")
    print("run 3_view_best.bat to replay the best gait")


if __name__ == "__main__":
    main()

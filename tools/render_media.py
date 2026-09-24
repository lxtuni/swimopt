# -*- coding: utf-8 -*-
"""
Render the images used by the README: a still frame and an animated GIF of the
robot swimming, plus a screenshot of the control panel.

Usage:
    python tools/render_media.py              # both gaits, then the panel
    python tools/render_media.py --sim        # simulation media only
    python tools/render_media.py --panel      # control panel screenshot only
    python tools/render_media.py --panel --run 4000  # run a real search, then capture
    python tools/render_media.py --gait PATH/best.json --stem NAME [--lift]
                                              # one extra clip from any result file

Both gaits are rendered from the same camera so the two clips can be compared
directly: the hand-built demo gait and whatever is in results/best.json.

The simulation is rendered offscreen, so it needs no visible window and gives the
same frames on any machine. The panel screenshot does need a desktop session,
because it captures a real Tk window.
"""
import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulate import Swimmer, load_cfg   # noqa: E402
import view as view_mod                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")

WIDTH, HEIGHT = 960, 540
GIF_SECONDS = 5.0
GIF_FPS = 20
GIF_COLORS = 64          # the scene is mostly blues, so a small palette is enough


def _camera(data, trunk_id, body_len):
    """A three-quarter view framed on the robot, sized from its own body length."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.xpos[trunk_id]
    cam.lookat[2] -= 0.10 * body_len         # sit the waterline higher in frame
    # Wide enough that the legs stay in frame when the body pitches, which the
    # optimized gaits do a lot of.
    cam.distance = 3.0 * body_len
    cam.azimuth = 128.0
    cam.elevation = -14.0
    return cam


def render_sim(use_demo=False, stem=None, best_path=None, lift=False):
    """Render one gait. best_path picks a result file; lift renders under lift physics,
    which a gait optimized with hydro.lift on must be, to look the way it scored."""
    cfg = load_cfg(os.path.join(ROOT, "config.json"))
    cfg["hydro"]["lift"] = bool(lift)
    os.chdir(ROOT)                      # model and result paths in the config are relative
    sw = Swimmer(cfg["model"], cfg)

    if use_demo:
        x, label = view_mod.demo_params(sw.gait), "demo gait"
        stem = stem or "gait_demo"
    else:
        x, label = view_mod.load_gait(sw, ["config.json", best_path or "results/best.json"])
        stem = stem or "gait_optimized"
    x = sw.gait.expand(x)
    r = sw.rollout(x)
    print(f"[sim] {label}{' (lift physics)' if lift else ''}: speed {r['speed']:+.4f} m/s, "
          f"{r['bl_s']:.3f} BL/s, RMS heading {r['heading_rms']:.1f}, roll "
          f"{r['roll_rms']:.1f}, pitch {r['pitch_rms']:.1f} deg, {r['power']:.2f} W")

    # Enlarge the offscreen framebuffer before the renderer is created.
    model = sw.model
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)

    m, d = model, sw.data
    p = sw.gait.decode(x)
    mujoco.mj_resetData(m, d)
    dt = m.opt.timestep
    n_settle = int(sw.settle / dt)
    steps_per_frame = max(1, int(round(1.0 / (GIF_FPS * dt))))
    n_frames = int(GIF_SECONDS * GIF_FPS)

    os.makedirs(DOCS, exist_ok=True)
    frames = []
    with mujoco.Renderer(m, height=HEIGHT, width=WIDTH) as renderer:
        for _ in range(n_settle):       # let it float up before filming
            sw.hydro.apply(d)
            mujoco.mj_step(m, d)
        cam = _camera(d, sw.trunk_id, sw.body_len)
        for _ in range(n_frames):
            for _ in range(steps_per_frame):
                t = d.time - sw.settle
                d.ctrl[:] = sw.gait.ctrl(p, max(0.0, t))
                sw.hydro.apply(d)
                mujoco.mj_step(m, d)
            cam.lookat[:] = d.xpos[sw.trunk_id]      # follow the robot
            cam.lookat[2] -= 0.10 * sw.body_len
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render().copy())

    from PIL import Image
    still = Image.fromarray(frames[len(frames) // 3])
    still_path = os.path.join(DOCS, f"{stem}.png")
    still.save(still_path)
    print(f"[sim] wrote {os.path.relpath(still_path, ROOT)}")

    gif_path = os.path.join(DOCS, f"{stem}.gif")
    small = [Image.fromarray(f).resize((WIDTH // 2, HEIGHT // 2), Image.LANCZOS)
             for f in frames]
    # One shared palette across frames, otherwise each frame carries its own.
    palette = small[0].quantize(colors=GIF_COLORS, method=Image.MEDIANCUT)
    quant = [im.quantize(palette=palette, dither=Image.FLOYDSTEINBERG) for im in small]
    quant[0].save(gif_path, save_all=True, append_images=quant[1:],
                  duration=int(1000 / GIF_FPS), loop=0, optimize=True)
    size_mb = os.path.getsize(gif_path) / 1e6
    print(f"[sim] wrote {os.path.relpath(gif_path, ROOT)} ({size_mb:.1f} MB)")


def render_panel(budget=None, seed=None):
    """Screenshot the real control panel. Needs an interactive desktop session.

    With `budget`, it first drives a real optimization through the panel, so the
    convergence curve, the log and the parameter table all show one genuine run
    rather than an empty window.
    """
    import time
    import tkinter as tk
    from PIL import ImageGrab

    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    import ui

    root = tk.Tk()
    root.geometry("+40+40")
    app = ui.App(root)

    if budget:
        app.e_budget.delete(0, "end")
        app.e_budget.insert(0, str(budget))
        if seed is not None:
            app.e_seed.delete(0, "end")
            app.e_seed.insert(0, str(seed))
        app.cb_view.set(ui.VIEW_NONE)    # no viewer window to overlap the screenshot
        app._start()
        print(f"[panel] running {budget} evaluations through the panel ...")
        deadline = time.time() + 1800
        while time.time() < deadline:
            root.update()
            if app.lb_stat.cget("text") == "Finished":
                break
            time.sleep(0.05)
        print(f"[panel] run finished: {app.lb_conv.cget('text')}")

    app._show_best()                     # fill the table if a result exists
    root.update_idletasks()
    root.update()
    root.lift()
    root.attributes("-topmost", True)
    root.update()
    root.after(900, root.quit)           # give the window manager time to paint
    root.mainloop()

    x, y = root.winfo_rootx(), root.winfo_rooty()
    w, h = root.winfo_width(), root.winfo_height()
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    root.destroy()

    os.makedirs(DOCS, exist_ok=True)
    out = os.path.join(DOCS, "control_panel.png")
    img.save(out)
    print(f"[panel] wrote {os.path.relpath(out, ROOT)} ({img.width}x{img.height})")


def _arg(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--gait" in args:
        # One extra clip, e.g. a study result:
        #   --gait results_cmp/lift_limits_s2/best.json --stem gait_lift --lift
        render_sim(best_path=_arg(args, "--gait"), stem=_arg(args, "--stem", "gait_extra"),
                   lift="--lift" in args)
        sys.exit(0)
    do_sim = "--panel" not in args
    do_panel = "--sim" not in args
    budget = int(_arg(args, "--run")) if "--run" in args else None
    seed = int(_arg(args, "--seed")) if "--seed" in args else None
    if do_sim:
        render_sim(use_demo=True)
        render_sim(use_demo=False)
    if do_panel:
        render_panel(budget, seed)

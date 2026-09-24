# -*- coding: utf-8 -*-
"""
swimopt control panel -- model import, model switching, search-space selection,
optimization and playback, all in one window.

Run:  python ui.py      (or double-click 8_control_panel.bat)
"""
import os
import re
import sys
import json
import glob
import queue
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

# High-DPI displays on Windows: without this the text renders blurry.
try:
    from ctypes import windll
    try:
        windll.shcore.SetProcessDpiAwareness(2)      # per-monitor v2
    except Exception:
        windll.user32.SetProcessDPIAware()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = sys.executable

RUN_CFG = "_ui_run.json"          # config written from the panel, fed to the workers

# What the panel reads from the optimizers' output. optimize.py prints lines in
# exactly these shapes; tests/test_optimize.py checks the two stay in step.
CANDIDATE_RE = re.compile(r"#\s*(\d+).*?fitness\s*([-+0-9.]+)")
GENERATION_RE = re.compile(r"evals\s+(\d+).*?best\s*([-+0-9.]+)")
SIGMA_RE = re.compile(r"sigma\s+([0-9.]+)")

# Playback modes, shown in the dropdown.
VIEW_EVERY = "every candidate"
VIEW_FIFTH = "one in five"
VIEW_BEST = "new records only"
VIEW_NONE = "no window (fastest)"


class App:
    def __init__(self, root):
        self.root = root
        root.title("swimopt control panel -- gait optimization for a paddling quadruped")
        self.proc = None
        self.q = queue.Queue()
        self.hist = []            # (eval index, best fitness so far)
        self.cfg_path = "config.json"
        self._setup_fonts()
        self._build()
        self._refresh_models()
        self._load_cfg_into_ui()
        for e in (self.e_lh, self.e_lr, self.e_lp, self.e_wene,
                  self.e_amax, self.e_f0, self.e_f1, self.e_omax):
            e.bind("<FocusOut>", lambda ev: self._update_dim())
            e.bind("<Return>", lambda ev: self._update_dim())
        self._size_to_content()
        self._stopped = False
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(300, self._pump)

    def _kill_run(self):
        """End the running job and every worker process it started."""
        if not (self.proc and self.proc.poll() is None):
            return
        if os.name == "nt":
            # terminate() would only kill the parent; its worker pool would linger.
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)],
                           capture_output=True)
        else:
            self.proc.terminate()

    def _on_close(self):
        self._kill_run()
        self.root.destroy()

    def _size_to_content(self):
        """Size the window to what the widgets actually need.

        A fixed size clips the top rows on a high-DPI display, where the same
        layout occupies more pixels.
        """
        self.root.update_idletasks()
        w = max(self.root.winfo_reqwidth(), 1340)
        h = max(self.root.winfo_reqheight(), 920)
        w = min(w, self.root.winfo_screenwidth() - 80)
        h = min(h, self.root.winfo_screenheight() - 120)
        self.root.geometry(f"{w}x{h}")
        self.root.minsize(min(w, 1100), min(h, 760))

    def _setup_fonts(self):
        """One UI font, one monospace font for numeric tables, sized to stay crisp."""
        avail = set(tkfont.families())
        fam = next((f for f in ("Segoe UI", "Helvetica", "Arial") if f in avail),
                   "TkDefaultFont")
        mono = next((f for f in ("Consolas", "Courier New") if f in avail), "TkFixedFont")
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(family=fam, size=11)
            except Exception:
                pass
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure(".", font=(fam, 11))
        st.configure("TLabelframe.Label", font=(fam, 11, "bold"), foreground="#1a3c6e")
        st.configure("TButton", font=(fam, 11), padding=5)
        st.configure("Treeview", font=(mono, 11), rowheight=26)
        st.configure("Treeview.Heading", font=(fam, 11, "bold"))
        self.font_mono = (mono, 11)
        self.font_big = (fam, 12, "bold")

    # ---------------- layout ----------------
    def _build(self):
        pad = dict(padx=6, pady=4)

        # === 1. model ===
        f1 = ttk.LabelFrame(self.root, text=" 1. Model ")
        f1.pack(fill="x", **pad)
        ttk.Label(f1, text="Active model:").pack(side="left", padx=6)
        self.cb_model = ttk.Combobox(f1, width=34, state="readonly")
        self.cb_model.pack(side="left", padx=4)
        self.cb_model.bind("<<ComboboxSelected>>", lambda e: self._switch_model())
        ttk.Button(f1, text="Import URDF...", command=self._import_urdf).pack(side="left", padx=4)
        ttk.Button(f1, text="Refresh list", command=self._refresh_models).pack(side="left", padx=4)
        self.lb_model = ttk.Label(f1, text="", foreground="#25507a")
        self.lb_model.pack(side="left", padx=12)

        # === 2. search space ===
        f2 = ttk.LabelFrame(self.root, text=" 2. Search space -- which parameters the optimizer varies ")
        f2.pack(fill="x", **pad)
        self.v_freq = tk.BooleanVar(value=True)
        self.v_duty = tk.BooleanVar(value=True)
        self.v_amp = tk.BooleanVar(value=True)
        self.v_phase = tk.BooleanVar(value=True)
        self.v_off = tk.BooleanVar(value=True)
        for txt, var in [("Frequency f", self.v_freq),
                         ("Duty cycle (power stroke %)", self.v_duty),
                         ("Amplitude A", self.v_amp),
                         ("Phase phi (= the gait)", self.v_phase),
                         ("Offset", self.v_off)]:
            ttk.Checkbutton(f2, text=txt, variable=var,
                            command=self._update_dim).pack(side="left", padx=8)
        ttk.Label(f2, text="   When phase is frozen, pin it to:").pack(side="left")
        self.cb_preset = ttk.Combobox(f2, width=16, state="readonly",
                                      values=["diag (diagonal)", "fb (front-back)",
                                              "lr (left-right)", "wave (travelling)",
                                              "inphase (all together)"])
        self.cb_preset.current(0)
        self.cb_preset.pack(side="left", padx=4)
        self.cb_preset.bind("<<ComboboxSelected>>", lambda e: self._update_dim())
        self.lb_dim = ttk.Label(f2, text="", foreground="#a0410d", font=self.font_big)
        self.lb_dim.pack(side="left", padx=14)

        # === 3. run settings ===
        f3 = ttk.LabelFrame(self.root, text=" 3. Run settings ")
        f3.pack(fill="x", **pad)
        self.e_budget = self._entry(f3, "Evaluations", "250", 6)
        self.e_time = self._entry(f3, "Seconds per rollout", "8.0", 6)
        self.e_pop = self._entry(f3, "Population size", "10", 5)
        self.e_seed = self._entry(f3, "Random seed", "1", 5)
        ttk.Label(f3, text="  Playback:").pack(side="left")
        self.cb_view = ttk.Combobox(f3, width=18, state="readonly",
                                    values=[VIEW_EVERY, VIEW_FIFTH, VIEW_BEST, VIEW_NONE])
        self.cb_view.set(VIEW_BEST)
        self.cb_view.pack(side="left", padx=4)
        self.v_rt = tk.BooleanVar(value=False)
        ttk.Checkbutton(f3, text="Real-time speed", variable=self.v_rt).pack(side="left", padx=6)

        # === 4. objective and search ranges ===
        f4 = ttk.LabelFrame(self.root, text=" 4. Objective -- what counts as swimming well ")
        f4.pack(fill="x", **pad)
        r1 = ttk.Frame(f4)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Maximise speed, keeping RMS   heading within").pack(
            side="left", padx=(8, 2))
        self.e_lh = ttk.Entry(r1, width=5)
        self.e_lh.insert(0, "10")
        self.e_lh.pack(side="left")
        ttk.Label(r1, text="deg,  roll within").pack(side="left", padx=2)
        self.e_lr = ttk.Entry(r1, width=5)
        self.e_lr.insert(0, "10")
        self.e_lr.pack(side="left")
        ttk.Label(r1, text="deg,  pitch within").pack(side="left", padx=2)
        self.e_lp = ttk.Entry(r1, width=5)
        self.e_lp.insert(0, "15")
        self.e_lp.pack(side="left")
        ttk.Label(r1, text="deg;   power weight").pack(side="left", padx=2)
        self.e_wene = ttk.Entry(r1, width=7)
        self.e_wene.insert(0, "0.0")
        self.e_wene.pack(side="left")
        ttk.Label(r1, text="per W").pack(side="left", padx=2)
        r1b = ttk.Frame(f4)
        r1b.pack(fill="x", pady=2)
        ttk.Label(r1b, text="Presets:").pack(side="left", padx=(8, 4))
        for txt, lh, lr, lp, we in [("Straight + level", "10", "10", "15", "0"),
                                    ("Strict", "5", "5", "8", "0"),
                                    ("Power-thrifty", "10", "10", "15", "0.02"),
                                    ("Fastest (no limits)", "180", "180", "180", "0")]:
            ttk.Button(r1b, text=txt, width=20,
                       command=lambda a=lh, b=lr, c=lp, e=we: self._set_w(a, b, c, e)).pack(
                           side="left", padx=2)
        r2 = ttk.Frame(f4)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="Search ranges:   frequency f").pack(side="left", padx=(8, 2))
        self.e_f0 = ttk.Entry(r2, width=5)
        self.e_f0.insert(0, "0.2")
        self.e_f0.pack(side="left")
        ttk.Label(r2, text="to").pack(side="left", padx=2)
        self.e_f1 = ttk.Entry(r2, width=5)
        self.e_f1.insert(0, "2.5")
        self.e_f1.pack(side="left")
        ttk.Label(r2, text="Hz     max amplitude A").pack(side="left", padx=(10, 2))
        self.e_amax = ttk.Entry(r2, width=5)
        self.e_amax.insert(0, "0.9")
        self.e_amax.pack(side="left")
        ttk.Label(r2, text="rad  (lower it if runs diverge)").pack(side="left", padx=2)
        ttk.Label(r2, text="   offset range +/-").pack(side="left", padx=(10, 2))
        self.e_omax = ttk.Entry(r2, width=5)
        self.e_omax.insert(0, "0.5")
        self.e_omax.pack(side="left")
        ttk.Label(r2, text="rad").pack(side="left")
        self.lb_fit = ttk.Label(f4, text="", foreground="#a0410d")
        self.lb_fit.pack(anchor="w", padx=10, pady=(2, 4))

        # === buttons ===
        f5 = ttk.Frame(self.root)
        f5.pack(fill="x", **pad)
        self.b_demo = ttk.Button(f5, text="Play demo gait", command=self._demo)
        self.b_demo.pack(side="left", padx=4)
        self.b_run = ttk.Button(f5, text="Start optimization", command=self._start)
        self.b_run.pack(side="left", padx=4)
        self.b_stop = ttk.Button(f5, text="Stop", command=self._stop, state="disabled")
        self.b_stop.pack(side="left", padx=4)
        self.b_best = ttk.Button(f5, text="Play best result", command=self._play_best)
        self.b_best.pack(side="left", padx=4)
        self.lb_stat = ttk.Label(f5, text="Ready", foreground="#25507a", font=self.font_big)
        self.lb_stat.pack(side="left", padx=20)

        # === middle: parameter table on the left, convergence plot on the right ===
        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, **pad)

        fl = ttk.LabelFrame(mid, text=" Best gait so far (per-actuator motion) ")
        fl.pack(side="left", fill="both", expand=True, padx=(0, 4))
        ttk.Label(fl, text="angle = offset + A1 sin(2 pi f t + phi1) + A2 sin(4 pi f t + phi2)",
                  foreground="#555", wraplength=560).pack(anchor="w", padx=8, pady=(4, 0))
        ttk.Label(fl, text="offset = centre of the swing | phase = when the stroke starts, "
                           "and the phase differences between legs are the gait | "
                           "second harmonic = flipper feathering",
                  foreground="#888", wraplength=560).pack(anchor="w", padx=8, pady=(0, 4))
        cols = ("joint", "off", "a1", "p1", "a2", "p2")
        self.tv = ttk.Treeview(fl, columns=cols, show="headings", height=13)
        for c, t, w in zip(cols,
                           ["Actuator", "Offset", "A (1st)", "Phase (1st)",
                            "A (2nd)", "Phase (2nd)"],
                           [150, 80, 90, 100, 90, 100]):
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor="center")
        self.tv.pack(fill="both", expand=True, padx=4, pady=4)
        self.lb_freq = ttk.Label(fl, text="Frequency: --", font=self.font_big,
                                 foreground="#0d4d2a")
        self.lb_freq.pack(anchor="w", padx=8, pady=(0, 6))

        fr = ttk.LabelFrame(mid, text=" Convergence (curve flattens + sigma shrinks = converged) ")
        fr.pack(side="left", fill="both", expand=True, padx=(4, 0))
        self.cv = tk.Canvas(fr, bg="white", height=250)
        self.cv.pack(fill="both", expand=True, padx=6, pady=6)
        self.lb_conv = ttk.Label(fr, text="evals 0 | best -- | sigma --", font=self.font_big,
                                 foreground="#0d4d2a")
        self.lb_conv.pack(anchor="w", padx=8, pady=(0, 6))

        # === log ===
        f6 = ttk.LabelFrame(self.root, text=" Run log ")
        f6.pack(fill="both", expand=True, **pad)
        self.txt = tk.Text(f6, height=9, font=self.font_mono)
        self.txt.pack(fill="both", expand=True, padx=4, pady=4)

    def _set_w(self, lh, lr, lp, we):
        for entry, value in ((self.e_lh, lh), (self.e_lr, lr), (self.e_lp, lp),
                             (self.e_wene, we)):
            entry.delete(0, "end")
            entry.insert(0, value)
        self._update_dim()

    def _fit_hint(self):
        try:
            lh, lr, lp = (float(e.get()) for e in (self.e_lh, self.e_lr, self.e_lp))
            we = float(self.e_wene.get())
        except ValueError:
            return "(one of the limits is not a number)"
        if min(lh, lr, lp) <= 0:
            return "(limits must be positive)"
        off = [n for n, v in (("heading", lh), ("roll", lr), ("pitch", lp)) if v >= 90]
        parts = ["Now: fastest gait whose RMS heading, roll and pitch stay inside the limits"]
        if off:
            parts.append(f"{' and '.join(off)} effectively unlimited -- expect "
                         f"{'veering' if 'heading' in off else 'rolling or rocking'}")
        if we > 0:
            parts.append(f"minus {we} BL/s per watt of mean power")
        return "; ".join(parts)

    def _entry(self, parent, label, default, width):
        ttk.Label(parent, text=f"  {label}:").pack(side="left")
        e = ttk.Entry(parent, width=width)
        e.insert(0, default)
        e.pack(side="left", padx=2)
        return e

    # ---------------- config read/write ----------------
    def _load_cfg_into_ui(self):
        try:
            sys.path.insert(0, HERE)
            from simulate import load_cfg
            from simulate import DEFAULT_LIMITS
            c = load_cfg(self.cfg_path)
            lim = dict(DEFAULT_LIMITS, **c.get("limits", {}))
            for entry, value in [(self.e_budget, c.get("budget", 250)),
                                 (self.e_time, c.get("sim_time", 8.0)),
                                 (self.e_pop, c.get("popsize", 10)),
                                 (self.e_seed, c.get("seed", 1)),
                                 (self.e_lh, lim["heading_deg"]),
                                 (self.e_lr, lim["roll_deg"]),
                                 (self.e_lp, lim["pitch_deg"]),
                                 (self.e_wene, c.get("w_energy", 0.0))]:
                entry.delete(0, "end")
                entry.insert(0, str(value))
            g = c.get("gait", {})
            fr = g.get("freq_range", [0.2, 2.5])
            ar = g.get("amp_range", [0.0, 0.9])
            orr = g.get("offset_range", [-0.5, 0.5])
            for entry, value in [(self.e_f0, fr[0]), (self.e_f1, fr[1]),
                                 (self.e_amax, ar[1]), (self.e_omax, abs(orr[1]))]:
                entry.delete(0, "end")
                entry.insert(0, str(value))
            m = c.get("model", "")
            if m in self.cb_model["values"]:
                self.cb_model.set(m)
        except Exception as e:
            self._log(f"[!] could not read config: {e}")
        self._update_dim()

    def _make_run_cfg(self):
        """Write the panel's settings to _ui_run.json, based on config.json."""
        from simulate import load_cfg
        c = load_cfg(self.cfg_path)
        c["budget"] = int(self.e_budget.get())
        c["sim_time"] = float(self.e_time.get())
        c["popsize"] = int(self.e_pop.get())
        c["seed"] = int(self.e_seed.get())
        c["limits"] = {"heading_deg": float(self.e_lh.get()),
                       "roll_deg": float(self.e_lr.get()),
                       "pitch_deg": float(self.e_lp.get())}
        c["w_energy"] = float(self.e_wene.get())
        g = c.setdefault("gait", {})
        g["freq_range"] = [float(self.e_f0.get()), float(self.e_f1.get())]
        g["amp_range"] = [0.0, float(self.e_amax.get())]
        o = abs(float(self.e_omax.get()))
        g["offset_range"] = [-o, o]
        g["optimize"] = {"freq": self.v_freq.get(), "duty": self.v_duty.get(),
                         "amp": self.v_amp.get(), "phase": self.v_phase.get(),
                         "offset": self.v_off.get()}
        g["preset_phase"] = self.cb_preset.get().split()[0]
        with open(RUN_CFG, "w", encoding="utf-8") as fh:
            json.dump(c, fh, ensure_ascii=False, indent=1)
        return RUN_CFG

    def _update_dim(self):
        try:
            from simulate import Swimmer, load_cfg
            cfg = load_cfg(self._make_run_cfg())
            sw = Swimmer(cfg["model"], cfg)
            self.lb_dim.config(text=f"optimizing {sw.gait.dim_opt} / {sw.gait.dim} dims")
            self.lb_model.config(text=f"{sw.model.nu} actuators | {sw.model.nbody-1} bodies")
            self.lb_fit.config(text=self._fit_hint())
        except Exception as e:
            self.lb_dim.config(text=f"(load failed: {str(e)[:40]})")

    # ---------------- models ----------------
    def _refresh_models(self):
        files = sorted(f.replace("\\", "/") for f in glob.glob("robots/*.xml"))
        self.cb_model["values"] = files
        if files and not self.cb_model.get():
            self.cb_model.current(0)

    def _switch_model(self):
        p = self.cb_model.get()
        r = subprocess.run([PY, "set_model.py", p], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        self._log(r.stdout.strip() or r.stderr.strip())
        self._update_dim()

    def _import_urdf(self):
        f = filedialog.askopenfilename(title="Select a URDF",
                                       filetypes=[("URDF", "*.urdf"), ("All files", "*.*")])
        if not f:
            return
        name = os.path.splitext(os.path.basename(f))[0].lower()
        out = f"robots/{name}.xml"
        self._log(f"importing {f} -> {out} ...")
        r = subprocess.run([PY, "import_model.py", f, out, "--drive", "2.1,1.1",
                            "--armature", "3e-4", "--damping", "0.02",
                            "--kp", "15", "--force", "3"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self._log(r.stdout.strip() or r.stderr.strip())
        self._refresh_models()
        if os.path.exists(out):
            self.cb_model.set(out)
            self._switch_model()

    # ---------------- running ----------------
    def _spawn(self, cmd, tag):
        if self.proc and self.proc.poll() is None:
            messagebox.showinfo("swimopt", "A run is already in progress. Stop it first.")
            return
        self._stopped = False
        self.txt.delete("1.0", "end")
        self.hist.clear()
        self._draw_curve()
        self._log("$ " + " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.b_run.config(state="disabled")
        self.b_stop.config(state="normal")
        self.lb_stat.config(text=tag)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.proc.stdout:
            self.q.put(line.rstrip())
        self.q.put(None)

    def _start(self):
        p = self._make_run_cfg()
        mode = self.cb_view.get()
        n = self.e_budget.get()
        if mode == VIEW_NONE:
            cmd = [PY, "-u", "optimize.py", p, n]
        else:
            cmd = [PY, "-u", "optimize_view.py", p, n]
            if mode == VIEW_FIFTH:
                cmd += ["--every", "5"]
            elif mode == VIEW_BEST:
                cmd += ["--best"]
            if self.v_rt.get():
                cmd += ["--rt"]
        self._spawn(cmd, "Optimizing...")

    def _demo(self):
        self._spawn([PY, "-u", "view.py", self._make_run_cfg(), "--demo"], "Playing demo gait")

    def _play_best(self):
        if not os.path.exists("results/best.json"):
            messagebox.showinfo("swimopt", "No results/best.json yet. Run an optimization first.")
            return
        self._spawn([PY, "-u", "view.py", self._make_run_cfg(), "results/best.json"],
                    "Playing best gait")

    def _stop(self):
        self._stopped = True
        self._kill_run()
        self.lb_stat.config(text="Stopped")
        self.b_run.config(state="normal")
        self.b_stop.config(state="disabled")

    # ---------------- log and plot ----------------
    def _log(self, s):
        if not s:
            return
        self.txt.insert("end", s + "\n")
        self.txt.see("end")

    def _pump(self):
        """Drain the worker's stdout and update the plot. Runs on the Tk thread.

        The patterns below must stay in step with what optimize.py and
        optimize_view.py print.
        """
        try:
            while True:
                line = self.q.get_nowait()
                if line is None:
                    self.lb_stat.config(text="Stopped" if self._stopped else "Finished")
                    self.b_run.config(state="normal")
                    self.b_stop.config(state="disabled")
                    self._show_best()
                    continue
                self._log(line)
                m = CANDIDATE_RE.search(line)
                if m:
                    ev, fit = int(m.group(1)), float(m.group(2))
                    best = max(fit, self.hist[-1][1] if self.hist else -9e9)
                    self.hist.append((ev, best))
                    self._draw_curve()
                gen_line = GENERATION_RE.search(line)
                sigma = SIGMA_RE.search(line)
                if gen_line or sigma:
                    sig = f"{float(sigma.group(1)):.4f}" if sigma else "--"
                    ev = self.hist[-1][0] if self.hist else 0
                    bf = f"{self.hist[-1][1]:+.4f}" if self.hist else "--"
                    self.lb_conv.config(text=f"evals {ev} | best {bf} | sigma {sig}")
                if "NEW BEST" in line:
                    self._show_best()
        except queue.Empty:
            pass
        self.root.after(250, self._pump)

    def _draw_curve(self):
        c = self.cv
        c.delete("all")
        W = c.winfo_width() or 500
        H = c.winfo_height() or 250
        if len(self.hist) < 2:
            c.create_text(W / 2, H / 2, text="waiting for data...", fill="#888")
            return
        pts_ok = [(x, y) for x, y in self.hist if y > -1.0]      # drop diverged runs
        if len(pts_ok) < 2:
            c.create_text(W / 2, H / 2,
                          text="waiting for valid data (early runs often diverge)",
                          fill="#888")
            return
        xs = [p[0] for p in pts_ok]
        ys = [p[1] for p in pts_ok]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        span = max(y1 - y0, 1e-3)
        y0 -= span * 0.1
        y1 += span * 0.1
        # Left margin follows the actual label width, otherwise the axis values are
        # clipped on a high-DPI display where the same text is wider.
        label_w = tkfont.Font(font=("", 9)).measure(f"{y1:+.3f}")
        mL, mR, mT, mB = label_w + 16, 16, 14, 34

        def px(x):
            return mL + (x - x0) / (x1 - x0 + 1e-9) * (W - mL - mR)

        def py(y):
            return H - mB - (y - y0) / (y1 - y0) * (H - mT - mB)

        c.create_line(mL, H - mB, W - mR, H - mB, fill="#999")
        c.create_line(mL, mT, mL, H - mB, fill="#999")
        pts = []
        for x, y in pts_ok:
            pts += [px(x), py(y)]
        c.create_line(*pts, fill="#c81e3c", width=3)
        c.create_oval(px(xs[-1]) - 4, py(ys[-1]) - 4, px(xs[-1]) + 4, py(ys[-1]) + 4,
                      fill="#c81e3c", outline="")
        c.create_text(W / 2, H - 12, text="evaluations", fill="#555", font=("", 9))
        c.create_text(mL - 8, mT + 8, text=f"{y1:+.3f}", anchor="e", fill="#555", font=("", 9))
        c.create_text(mL - 8, H - mB, text=f"{y0:+.3f}", anchor="e", fill="#555", font=("", 9))
        c.create_text(W - mR, H - mB + 15, text=f"{x1}", anchor="e", fill="#555", font=("", 9))
        c.create_text(mL + 6, mT + 8, text="fitness (higher is better)", anchor="w",
                      fill="#999", font=("", 9))

    def _best_swimmer(self, cfg_file):
        """A Swimmer for the current run config, rebuilt only when that file changes.

        This runs on every new record during a search, and rebuilding the model each
        time made the window stutter.
        """
        stamp = os.path.getmtime(cfg_file)
        cached = getattr(self, "_sw_cache", None)
        if cached and cached[0] == (cfg_file, stamp):
            return cached[1]
        from simulate import Swimmer, load_cfg
        c = load_cfg(cfg_file)
        sw = Swimmer(c["model"], c)
        self._sw_cache = ((cfg_file, stamp), sw)
        return sw

    def _show_best(self):
        try:
            import numpy as np
            with open("results/best.json", encoding="utf-8") as fh:
                b = json.load(fh)
            sw = self._best_swimmer(RUN_CFG if os.path.exists(RUN_CFG) else self.cfg_path)
            p = sw.gait.decode(b["x"])
            ok = b.get("feasible")
            self.lb_freq.config(text=f"{p['freq']:.2f} Hz   "
                                     f"{b.get('speed', 0):+.4f} m/s   RMS heading "
                                     f"{b.get('heading_rms', 0):.1f}, roll "
                                     f"{b.get('roll_rms', 0):.1f}, pitch "
                                     f"{b.get('pitch_rms', 0):.1f} deg   "
                                     f"{b.get('power', 0):.1f} W   "
                                     + ("within limits" if ok else
                                        "OVER LIMIT" if ok is not None else ""))
            for it in self.tv.get_children():
                self.tv.delete(it)
            for i, nm in enumerate(sw.gait.names):
                off = p["offs"][sw.gait._off_slot[i]]
                a1 = p["A"][0][sw.gait._amp_slot[i]]
                p1 = np.degrees(p["PH"][0][i])
                a2 = p["A"][1][sw.gait._amp_slot[i]] if sw.gait.H > 1 else 0.0
                p2 = np.degrees(p["PH"][1][i]) if sw.gait.H > 1 else 0.0
                self.tv.insert("", "end",
                               values=(nm, f"{off:+.3f}", f"{a1:.3f}", f"{p1:6.1f}",
                                       f"{a2:.3f}", f"{p2:6.1f}"))
        except Exception as e:
            self._log(f"[!] could not read the best result: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.15)
    except Exception:
        pass
    App(root)
    root.mainloop()

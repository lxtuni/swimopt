# swimopt

**Work out how a legged robot should swim — automatically.**

You hand it a robot model. It figures out how the robot should move its legs to swim
as fast, as straight, or as efficiently as you ask for, by simulating the paddling and
searching the space of possible strokes.

<table>
<tr>
<td width="50%"><img src="docs/gait_demo.gif" alt="The toy quadruped swimming with a hand-built gait"></td>
<td width="50%"><img src="docs/gait_optimized.gif" alt="The same robot swimming with the optimized gait"></td>
</tr>
<tr>
<td><b>Hand-built gait</b><br>0.025 m/s &middot; 0.11 body lengths/s</td>
<td><b>After 250 CMA-ES evaluations</b><br>0.084 m/s &middot; 0.37 body lengths/s</td>
</tr>
</table>

Same robot, same water, same three minutes of searching. The right-hand gait is
3.4 times faster — and it also rolls hard, which is a real finding about the objective
rather than a rendering artefact. See [Status](#status).

A Chinese translation of the user guide is available in [README.zh-CN.md](README.zh-CN.md).

---

## What it actually does

Give it any MuJoCo or URDF robot. It then:

1. **Puts it in water.** A Morison-type hydrodynamic model is attached to every link —
   buoyancy, direction-dependent drag, added mass, viscous damping — with coefficients
   assigned by matching link names against rules in a config file.
2. **Gives every joint a stroke to follow.** Each actuator gets a two-harmonic Fourier
   series with its own amplitude, phase and mean angle, plus a duty-cycle warp so the
   power stroke can be faster than the recovery stroke.
3. **Searches for the best stroke.** CMA-ES varies those parameters, scores each
   candidate by an explicit objective, and keeps the best.

The output is a set of gait parameters you can replay, inspect per joint, or send to
real servos.

**Why bother.** Published paddling-robot studies usually hand-pick two or three gaits —
diagonal, front–back, wave — and compare them. The real space is much larger:
frequency, per-joint amplitude, phase offsets, mean angles, harmonic content and duty
cycle all interact. swimopt searches that space instead of sampling three points in it.

**Swapping robots touches one line.** The physics and control layers never mention a
specific robot, so a new machine means a new model file and a new `"model"` entry, not
a code change. That matters while the CAD is still moving.

Built for the thesis *"Hydrodynamic Modeling and Trajectory Optimization of Paddling
Propulsion for a Quadruped Robot."*

---

## Try it

Requires Python 3.10–3.12.

```bash
git clone https://github.com/lxtuni/swimopt.git
cd swimopt
pip install -r requirements.txt
```

```bash
python view.py     config.json --demo    # watch a hand-built gait swim
python optimize.py config.json 250       # search for a better one (~3 min)
python view.py     config.json           # watch what it found
```

On Windows, double-click the numbered launchers instead — `0_setup_env.bat` builds a
local virtualenv, then `1_demo_gait.bat` through `8_control_panel.bat` run the same
steps in order.

Results land in `results/` as `best.json`, `log.csv` and `convergence.json`.

---

## The control panel

`python ui.py`, or double-click `8_control_panel.bat`. Everything above, without the
command line.

![The swimopt control panel after a 250-evaluation run](docs/control_panel.png)

The four numbered sections are the whole workflow:

1. **Model** — pick a robot from `robots/`, or import a URDF and have it converted,
   actuated and activated in one click.
2. **Search space** — tick what the optimizer is allowed to vary. Unticking freezes a
   parameter, and the dimension count on the right updates live. Freezing phase and
   offset collapses the 35-dimensional search to 8, which is how you compare a fixed
   gait family (diagonal against front–back against wave) at its own best frequency
   and amplitude.
3. **Run settings** — budget, rollout length, population, seed, and how much of the
   search you want to watch happen.
4. **Objective** — the weights that decide what "swimming well" means, with presets
   for *fastest*, *fast and straight*, *strictly straight* and *power-thrifty*.

Below them: the best gait so far, one row per actuator, next to the convergence curve
and the CMA-ES step size.

---

## Architecture

```
robots/*.xml     1 model layer       the only thing that changes per robot
hydro.py         2 physics layer     robot-agnostic Morison hydrodynamics (vectorized)
gait.py          3 control layer     robot-agnostic Fourier + duty-cycle gait
simulate.py      --  glue: one rollout gives one score
optimize.py      4 optimization      CMA-ES
optimize_view.py 4 optimization      same, with live rendering
ui.py            Tkinter control panel
import_model.py  URDF to MJCF importer (portable meshes, automatic actuators)
set_model.py     switch the active robot
view.py          replay a gait or the best result
config.json      every parameter, with inline comments
tools/           regenerate the images in this README
```

---

## The physics

Per-link quasi-steady Morison-type force, scaled by immersion fraction `f`:

```
F = rho*g*V*f            buoyancy
  + 0.5*rho*Cd*A*v|v|*f  quadratic drag, anisotropic (per-axis Cd)
  + Ca*rho*V*(dv/dt)     added mass
  + cv*v*f               linear viscous
```

- **Partial immersion** comes from each geom's true vertical extent under its current
  orientation, not a bounding sphere, so a flipper rolling at the surface gets the
  right `f`.
- **Anisotropic drag is what produces thrust at all.** The paddle face has a large
  `Cd`, the thin edge a small one, and the difference over a stroke cycle is the net
  force.
- **Added mass** uses the "mass trick": add `m_a = Ca*rho*V` to the body's inertial
  mass and cancel the resulting extra weight with a constant upward force. It is an
  isotropic approximation, but it is numerically unconditionally stable, which the
  explicit `dv/dt` form is not.
- **Not modelled:** turbulence, vortex shedding, wake interaction, free-surface waves.
  These matter for violent manoeuvres; for steady paddling their mean effect is
  absorbed into experimentally calibrated coefficients. Same modelling level as the
  beaver-robot literature this builds on.

Links are matched to coefficients by **name pattern** in `config.json`, so a new robot
needs no code change — just a rule like
`{ "match": "flip", "cd": [2.2, 0.10, 0.10], "ca": 1.0 }`.

---

## The gait parameterization

```
q_i(t) = offset_i + sum_h  A_{i,h} * sin(2*pi*h*f*t + phi_{i,h})       h = 1, 2
```

plus a duty-cycle phase warp that lets the power stroke occupy a different fraction of
the cycle than the recovery stroke.

**The second harmonic is not optional.** With a first harmonic only, two legs driven in
antiphase produce exactly opposite thrust, all four legs cancel, and net thrust is
identically zero. A second harmonic is invariant under a phase shift of pi, which is
the mathematical counterpart of the real robot's passive flipper feathering — spread on
the power stroke, furl on recovery. It is the prerequisite for any net thrust at all.

Any parameter group can be frozen, from the panel or the config file.

---

## The optimizer

CMA-ES (Hansen & Ostermeier 2001), via the `cma` package. It is a **gradient-free
evolution strategy, not reinforcement learning** — it optimizes the parameters of a
fixed-form trajectory, not a state-feedback policy.

```
fitness = speed - w_yaw * yaw_rate - w_energy * mean_power
```

The weights are a ruler, not a result. Fitness values are not comparable across
different weight settings, so report the physical quantities — m/s, body lengths/s,
degrees, watts — never the fitness number.

Convergence means the fitness curve flattens **and** the step size sigma shrinks.
Re-running with a different seed should land nearby; if it does not, the budget was too
small.

---

## What is not in this repository

- The **BODY2 CAD model** (`robots/body2.xml` and its STL meshes) — lab property, and
  gitignored. `robots/toy_quad.xml` is a small open four-legged swimmer that exercises
  every feature and is enough to reproduce everything here.
- Reference papers — copyrighted.
- `mjenv/` — the machine-specific virtualenv.

To use your own robot:

```bash
python import_model.py my_robot.urdf robots/mine.xml --drive "2.1,1.1"
python set_model.py
```

The importer copies meshes locally and rewrites paths, so the model stays portable
across machines.

Note that URDF cannot express closed kinematic loops. A parallelogram or five-bar leg
imports as an open tree and produces no thrust; close it manually with MuJoCo
`<equality><connect>` after import.

---

## Status

Validated end to end on `toy_quad`. One run of `python optimize.py config.json 250`,
seed 1, MuJoCo 3.14, 250 rollouts in about three minutes on a laptop CPU:

| | hand-built gait | optimized gait |
|---|---|---|
| forward speed | 0.025 m/s | **0.084 m/s** |
| body lengths per second | 0.11 | **0.37** |
| yaw drift over 8 s | 0.2 deg | 4.7 deg |
| pitch amplitude | 19 deg | 38 deg |
| roll amplitude | 0.6 deg | **62 deg** |
| mean mechanical power | 70 W | **426 W** |

101 of the 250 rollouts diverged and were rejected by the speed guard, which is normal
for a first pass over a 35-dimensional space.

**Read that last column carefully.** The objective is
`speed - w_yaw*yaw_rate - w_energy*power`, and at the default weights
(`w_yaw = 0.3`, `w_energy = 0`) nothing penalizes roll, pitch or power. The optimizer
did exactly what it was asked and found a fast, violently rolling, power-hungry stroke.
The attitude figures are already computed every rollout but do not enter the score. If
you want a gait a real robot can hold, raise `w_energy` or add an attitude term — and
say which you used when you report a number.

That is the intended use of this tool: it makes the objective's blind spots visible
instead of hiding them behind three hand-picked gaits.

Rollouts are deterministic: identical parameters reproduce a result exactly, which the
replay path checks.

Coefficients are currently literature values. Experimental calibration against the
physical robot — coast-down, static draft, pendulum decay, biped paddling — is the next
step, and is what the accuracy ultimately rests on, not the choice of simulator.

## Regenerating the images

```bash
python tools/render_media.py --sim              # both gait clips, offscreen
python tools/render_media.py --panel --run 250  # real search, then screenshot
```

The simulation frames render offscreen, so they need no visible window and come out the
same on any machine. The panel screenshot drives a real optimization through the real
window, so it needs a desktop session and `pillow`.

---

## References

- Morison, J.R. et al. (1950). The force exerted by surface waves on piles. *J. Pet. Technol.* 2(5).
- Fossen, T.I. (1994). *Guidance and Control of Ocean Vehicles*. Wiley.
- Chen, G. et al. (2023). Modeling of swimming posture dynamics for a beaver-like robot. *Ocean Engineering*.
- Chen, G. et al. (2021). Hydrodynamic analysis of webbed feet. *Ocean Engineering* 234:109179.
- Hansen, N. & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation* 9(2):159–195.
- Lee, S. et al. (2025). Unified fluid–robot multiphysics simulation and optimization. arXiv:2506.05012. *(workflow reference; they use a gradient method, this project uses CMA-ES because MuJoCo is not differentiable)*

## License

MIT — see [LICENSE](LICENSE).

# swimopt

[![tests](https://github.com/lxtuni/swimopt/actions/workflows/tests.yml/badge.svg)](https://github.com/lxtuni/swimopt/actions/workflows/tests.yml)

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
<td><b>Hand-built gait</b><br>0.07 m/s &middot; 0.29 body lengths/s &middot; rocks 27° in pitch</td>
<td><b>Optimized bound, 2500 CMA-ES evaluations</b><br>0.47 m/s &middot; 1.98 body lengths/s &middot; straight and level</td>
</tr>
</table>

Same robot, same water, same 400 deg/s servos, about five minutes of searching on a
16-core laptop: 6.8 times faster, while staying inside 10 degrees RMS of heading and
roll and 15 of pitch. See [Status](#status) for how reliable that number is and what
it leans on.

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
python optimize.py config.json 4000      # search for a better one (a few minutes)
python view.py     config.json           # watch what it found
```

On Windows, double-click the numbered launchers instead — `0_setup_env.bat` builds a
local virtualenv, then `1_demo_gait.bat` through `8_control_panel.bat` run the same
steps in order.

Candidates are evaluated in parallel, one worker process per candidate up to your CPU
count; a parallel search returns exactly what a serial one would. Results land in
`results/`: `best.json` holds the full parameter vector, `run_config.json` the exact
configuration and library versions, plus `log.csv` and `convergence.json`.

To check an installation, or any change you make:

```bash
pip install -r requirements-dev.txt
python -m pytest
```

---

## The control panel

`python ui.py`, or double-click `8_control_panel.bat`. Everything above, without the
command line.

![The swimopt control panel after a 2500-evaluation bound search, seed 1](docs/control_panel.png)

The four numbered sections are the whole workflow:

1. **Model** — pick a robot from `robots/`, or import a URDF and have it converted,
   actuated and activated in one click.
2. **Search space** — the gait family (free, trot, pace, bound, walk, pronk; see
   [Gait families](#gait-families)) and what the optimizer is allowed to vary.
   Unticking freezes a parameter, and the dimension count on the right updates live.
3. **Run settings** — budget, rollout length, population, seed, and how much of the
   search you want to watch happen.
4. **Objective** — the goal (fastest gait, or least power at a required speed), the
   servos' no-load speed, and how straight and how level the gait must be, as RMS
   limits in degrees on heading, roll and pitch. Presets cover *straight and level*,
   *strict*, *power-thrifty*, and *no limits* for comparison.

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
tools/           studies and figures:
  describe_gait.py   what kind of gait a result is, and a gait diagram of it
  gait_families.py   every gait family, each at its own best stroke, several seeds
  pareto.py          least power at each required speed, drag vs lift
  compare_lift.py    drag vs lift at the fastest straight-and-level gait
  multi_seed.py      one search, several seeds, keep the best
  render_media.py    the clips and screenshots in this README
tests/           pytest suite: physics checks, servo model, families, parallel ==
                 serial, importer
```

There is one simulation loop, in `Swimmer.rollout`. The live viewer renders through
its callback rather than keeping a copy, so what you watch is exactly what is scored.

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
  beaver-robot literature this builds on. Every rollout reports the share of time a
  limb breaks the surface, so a gait that depends on what is not modelled shows up.

### Servos

A gait is only useful if real servos can drive it. Without a speed limit the best gait
found needed 2700 to 4200 degrees per second at the joints, five to ten times a hobby
servo, and on a 400 deg/s servo it does not move forward at all.

Each servo therefore has a DC motor's torque-speed curve: full stall torque at rest,
falling linearly to zero at the no-load speed `servo.max_speed_dps`.

```
tau = ts * v - (ts / wmax) * w        |v| <= 1, ts = stall torque, wmax = no-load speed
```

The second term is back-EMF, and it is simply viscous damping, so it is added to the
joint's damping and MuJoCo integrates it implicitly. Two simpler versions were tried
and failed; the tests keep them failed:

- **Limiting only the command.** When drag held a hip back, the servo fell behind and
  then whipped the leg forward at 1500 deg/s to catch up.
- **Rewriting the torque limit from the joint speed every step.** That is explicit
  damping. On the light flipper at a 2 ms step it went unstable and chattered at
  3000 deg/s, and the answer changed six-fold with the timestep.

With the implicit version the result is the same at 2 ms and 0.5 ms. In air a hip
peaks at 298 deg/s on a 300 deg/s servo. In water a joint can still be pushed past
the no-load speed by the flow or by the neighbouring link, as a real geared servo can
be back-driven; the tests check that the motor is then braking, never driving.

Every rollout reports the peak joint speed and the share of time a servo sits at its
torque limit.

### Drag-based and lift-based propulsion

The terms above are all **resistive**: every one is anti-parallel to a component of the
relative flow. A swimmer built only from them can only push water backwards — it
paddles. Real flippers also work as foils, generating force *across* the flow, and that
mechanism needs its own term.

`hydro.lift` adds it. It is **off by default**, so the base model stays purely
resistive unless you ask for lift:

```
Cl(alpha) = cl * sin(2*alpha)      alpha = angle between the flow and the plate's plane
L = 0.5*rho*Cl*A*|v|^2             perpendicular to the flow, in the (flow, normal) plane
```

This is the post-stall flat-plate model, which is the right level of description for a
paddle sweeping through large angles. The plate normal is taken to be the axis with the
largest `Cd`, so no extra geometry description is needed. Each link opts in with a `cl`
in its rule; anything not plate-like keeps `cl: 0`.

Why it changes conclusions: at small angle of attack lift grows like `alpha` while the
resistive terms grow like `alpha^2`. A fast, shallow, feathered sweep therefore
produces almost nothing in the resistive model and a great deal with lift enabled.

> **With `lift: false`, a lift-based gait cannot win a search here, because the
> mechanism that would let it win is not implemented.** A resistive-only result is a
> statement about drag-based paddling, not about optimal swimming in general. Say which
> setting produced any number you report.

Every rollout reports the signed forward impulse from the lift term and from the
resistive terms, so "is this gait lift-based or drag-based" is a measurement rather
than an impression. The split is checked against momentum conservation by the tests.

**`cl` is not calibrated.** 1.1 is the textbook flat-plate value. Results from it are
qualitative until the real flipper is measured.

#### Drag against lift at matched speed

The question "is lift-based swimming better" only has an answer at matched speed:
power rises steeply with speed, so comparing two fastest gaits compares two different
speeds. `tools/pareto.py` asks it directly. For each physics, each required speed and
each gait family, it searches for the **least mean power** that still reaches that
speed inside the straight-and-level limits, with a limb out of the water at most 5 % of
the time. The best over families is that physics' speed–power front.

```bash
python tools/pareto.py 2500 --seeds 2
```

`toy_quad`, 400 deg/s servos, 5 families × 2 seeds × 2500 evaluations per point:

![Least power at each speed, resistive-only against resistive plus lift](docs/pareto.png)

| required speed | resistive only | resistive + lift | lift / resistive |
|---|---|---|---|
| 0.10 m/s | 0.23 W, walk | 0.10 W, bound | 0.43 |
| 0.20 m/s | 1.04 W, bound | 0.55 W, pronk | 0.53 |
| 0.30 m/s | 3.10 W, trot | 0.97 W, bound | 0.31 |

<table>
<tr>
<td width="50%"><img src="docs/gait_lift.gif" alt="Least-power lift-based bound at 0.30 m/s"></td>
<td width="50%"><img src="docs/gait_optimized.gif" alt="Fastest resistive-only bound"></td>
</tr>
<tr>
<td><b>With lift: least power for 0.30 m/s</b><br>bound, 0.97 W; lift pushes, drag is pure loss</td>
<td><b>Resistive only: fastest gait</b><br>bound, 0.47 m/s at 11.6 W</td>
</tr>
</table>

What the data supports:

- **With lift in the model, every gait the search kept is lift-driven.** In all 22
  feasible lift-physics optima, across every family and speed, the lift term supplies
  all of the forward impulse (+0.46 to +2.5 N*s at the envelope) and the resistive
  terms are net negative. Given the choice, the optimizer never paddles.
- **With lift, the same speed costs less power at every speed tested**, by a factor of
  about two to three. Unlike the earlier cost-of-transport comparison, this is at
  matched speed, so "slower is cheaper" is not the explanation.
- **Each optimum depends on its physics.** Without lift, the lift-based optima fall to
  35 to 58 % of their speed. Re-scored with lift, none of the resistive optima stays
  straight and level: two swim faster but pitch past the limit, and the trot rolls over.
  Lift puts moments on the hull. That is also why fewer
  lift searches found a feasible gait at all, 22 of 30 against 27 of 30.

What it does not pin down:

- **The size of the gap.** The search is not converged. The second seed cut the
  resistive-only least power at 0.20 m/s from 2.99 to 1.04 W, and a few family searches
  found no feasible gait even where one plainly exists (bound reaches 0.47 m/s in the
  speed study). Both fronts are upper bounds on the true least power. The direction —
  lift below resistive at all three speeds, with both seeds' best — is the finding; the
  ratio is not.
- **Anything quantitative about a real flipper.** `cl` is uncalibrated, and the model
  has no unsteady effects (vortex shedding, wake capture) that real foils rely on and
  that could move the answer either way.

So for the claim that the optimal stroke is lift-based: in this model, yes, in the
sense that matters for a thesis. Whenever lift is available the optimizer uses it, and
at matched speed and attitude it needs markedly less power. Whether that survives
contact with the real flipper rests on measuring `cl`.

The earlier study, `tools/compare_lift.py` (fastest gait, drag against lift, with and
without the attitude limits), still runs; its published numbers predate the servo
model and are withdrawn.

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

**What the second harmonic is for.** It is invariant under a phase shift of pi, which
makes it the natural way to express flipper feathering — spread on the power stroke,
furl on recovery — the way a passive flipper does on the real robot.

An earlier version of this README claimed the second harmonic was the prerequisite for
any net thrust: with one harmonic, antiphase legs would push exactly opposite and
cancel. **That is wrong for this model, and was never tested.** The cancellation
argument only holds for strictly reciprocal motion. toy_quad drives three joints per
leg with independent phases, so even one harmonic sweeps the leg non-reciprocally.
Measured, 1600 evaluations each:

| harmonics | duty cycle | speed | straight and level |
|---|---|---|---|
| 1 | fixed at 50 %, a pure sinusoid | 0.156 m/s | roll 10.2°, just over |
| 1 | free, the search chose 26 % | 0.296 m/s | yes |

A fast power stroke with a slow recovery produces thrust on its own, because drag grows
with the square of speed. Whether a second harmonic is *necessary* depends on the leg:
it may well be for a leg with one actuated joint and a passive flipper, which is worth
testing on BODY2 rather than assuming.

Any parameter group can be frozen, from the panel or the config file.

### Gait families

By default every actuator has its own phase, 35 dimensions on toy_quad, so the search
can find coordinations no textbook names. Setting `gait.family` fixes the legs' timing
instead and searches only one phase per joint type, shared by all legs, 17 dimensions:

| family | also called | each leg's delay behind front-left, in strokes |
|---|---|---|
| `trot` | diag | FR 0.5, BL 0.5, BR 0 |
| `pace` | lr | FR 0.5, BL 0, BR 0.5 |
| `bound` | fb | FR 0, BL 0.5, BR 0.5 |
| `walk` | wave | lateral sequence: BR 0.25, FR 0.5, BL 0.75 |
| `pronk` | inphase | all together |

The delay is applied to time, before the duty-cycle warp, so in a trot the right leg
does exactly what the left leg did half a stroke earlier, even when the power stroke
is shorter than the recovery. Legs and joint types come from the model's geometry, not
from actuator names: a leg is the chain of bodies below one child of the floating
trunk, placed front or back and left or right by where it attaches, and a joint's type
is its depth down that chain. An earlier version read the names, and every family
silently became "all together" on a robot whose actuators were not called FL, FR, BL
and BR.

### Which gait family

`python tools/gait_families.py 2500 --seeds 3` searches every family at its own best
stroke: same budget, same seeds, everything but the legs' timing free. Resistive
physics, 400 deg/s servos, straight-and-level limits, mean ± sd over three seeds:

| family | speed m/s | stroke Hz | power W | cost of transport J/m |
|---|---|---|---|---|
| **bound** | **0.481 ± 0.034** | 2.49 | 12.8 | 27 |
| pace | 0.345 ± 0.064 | 2.26 | 10.5 | 32 |
| trot | 0.344 ± 0.048 | 1.82 | 11.5 | 34 |
| walk | 0.344 ± 0.098 | 2.00 | 12.0 | 38 |
| pronk | 0.265 ± 0.038 | 2.48 | 12.0 | 46 |
| free phases | 0.096 ± 0.046 | 0.86 | 1.6 | 15 |

All 18 winners are inside the limits. Two things stand out:

- **Bound wins, and by more than the seed spread.** Front legs together, back legs
  together, half a stroke apart: it is left–right symmetric, so heading and roll cost
  it nothing and all of its freedom goes into thrust. Its weakness is pitch, and all
  three bound winners sit at the 15-degree limit.
- **The free search loses to every family.** It can express every one of them, and its
  winners do resemble them (bound, pronk and trot, 7 to 19 % of a stroke off), but at
  2500 evaluations it has not found their good versions. That is a search failure, not
  a finding about coordination; it is why the family search is the default for studies.
  The free winners' low cost of transport comes with low speed and is not comparable.

Caveats: pronk seed 1 and trot seed 3 keep a limb out of the water 17 % and 49 % of the
time, outside what the model predicts; the other 16 at most 10 %. Set
`limits.surfacing` to exclude such gaits, as the speed–power study below does.

### Describing a gait

```bash
python tools/describe_gait.py results/best.json
```

It reads each leg's delay from the joint type that moves most, names the nearest family
and how far off it is, reports the validity checks, and draws the figure below from the
scoring rollout itself:

![Gait diagram of the current best gait](docs/gait_diagram.png)

The top panel is a gait diagram: each bar is a power stroke, defined physically as the
leg tip moving backward relative to the body.

---

## The optimizer

CMA-ES (Hansen & Ostermeier 2001), via the `cma` package. It is a **gradient-free
evolution strategy, not reinforcement learning** — it optimizes the parameters of a
fixed-form trajectory, not a state-feedback policy.

Two goals, set by `objective.mode`:

```
speed:  fastest gait,  fitness = speed in body lengths/s - w_energy * mean power
power:  least power,   fitness = -mean power,  subject to speed >= min_speed
```

Both are subject to the straight-and-level limits: RMS heading, roll and pitch, by
default 10, 10 and 15 degrees. Paddling rocks the body, so pitch is looser.

**Feasibility comes first.** A gait outside a limit, or short of the required speed,
scores in a band below every feasible gait, ranked by how far outside it is. No speed
buys its way back in. An earlier version subtracted one body length per second per
100 % of excess, which assumed nothing swims faster than one body length per second;
unconstrained gaits here reach five.

An earlier version used additive weights, `speed - w_yaw*yaw_rate - w_attitude*...`.
Weights like that only work at the speed scale they were tuned for. Once the model was
fixed and the legs could really move, speed swamped them and the best gait rolled 42
degrees and veered 44.

The objective is a ruler, not a result. Fitness values are not comparable across
different settings, so report the physical quantities — m/s, body lengths/s, degrees,
watts — never the fitness number.

**Budget.** The free gait space has 35 dimensions and is strongly multimodal. At 250
evaluations CMA-ES has barely started; at 2500 to 4000 with a population of 16 the
free search still ends 3 to 5 times slower than a family search that found the same
kind of gait (see [Which gait family](#which-gait-family)). A family search has 17
dimensions and is far more reliable; prefer it, and use the free search to look for
coordinations no family covers. Either way, treat a single run as a lower bound and
compare conditions over several seeds rather than trusting one number. Convergence means the fitness curve flattens **and** the step size sigma
shrinks.

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

Validated end to end on `toy_quad` with 400 deg/s servos: `gait.family` bound, 2500
rollouts, population 16, MuJoCo 3.14, about five minutes on 16 cores.

| | hand-built | optimized bound, seed 1 |
|---|---|---|
| forward speed | 0.070 m/s | **0.474 m/s** |
| body lengths per second | 0.29 | **1.98** |
| RMS heading / roll / pitch | 2.2 / 1.0 / **26.7** deg | 0.0 / 0.0 / 14.8 deg |
| mean mechanical power | 8.9 W | 11.6 W |
| stroke frequency | 1.2 Hz | 2.5 Hz |
| peak joint speed | 398 deg/s | 402 deg/s |
| limb out of the water | 0 % | 0.6 % |

A bound is left–right symmetric, so heading and roll stay at exactly zero; pitch is the
only limit it has to respect, and it sits on it.

**What that number leans on.** Read it as the model's answer, not the robot's:

- **The stroke frequency is on the search bound**, 2.5 Hz. The search would go faster
  still if allowed. Set `freq_range` to what the real servos sustain.
- **The servos are at their torque limit the whole time.** That is the torque-speed
  curve working as intended, but a real servo held at stall heats up; there is no
  thermal model. 14 of the 18 family-study winners do the same.
- **Most winners sit on one of the limits**: every bound and pace winner, and about
  two thirds of all 18 in the family study, end within a degree of the pitch or roll
  limit. The limits are shaping the answer, so report them with it.
- **Seed spread.** The three bound seeds reached 0.474, 0.518 and 0.450 m/s. Seed 2 is
  faster but keeps a limb out of the water 10 % of the time, which is outside what the
  model predicts; seed 1 is shown for that reason.

Run several seeds and keep the best valid one, or report the spread.

Everything is deterministic. The same seed reproduces a run bit for bit, the panel and
the command line reach identical results, and a parallel search returns exactly what a
serial one would; the test suite checks all three.

Coefficients are literature values and `cl` is a textbook flat-plate figure.
Experimental calibration against the physical robot — coast-down, static draft,
pendulum decay, biped paddling — is the next step, and is what the accuracy ultimately
rests on, not the choice of simulator.

### Correction

Numbers published in this README before September 2026 are withdrawn. They came from a
model whose joint limits were read in degrees instead of radians: every leg was pinned
to ±1.3 degrees, and the actuators spent about nine tenths of their power fighting the
limits. The same hand-built gait went from 0.025 to 0.092 m/s once fixed. The drag
versus lift comparison was rerun from scratch on the corrected model, with several
seeds, and the section above replaces the earlier one.

## Regenerating the images

```bash
python tools/render_media.py --sim              # both gait clips, offscreen
python tools/render_media.py --gait results_families/fb_s1/best.json --stem gait_optimized
python tools/render_media.py --panel --run 2500 --seed 1 --family fb   # real search, then screenshot
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

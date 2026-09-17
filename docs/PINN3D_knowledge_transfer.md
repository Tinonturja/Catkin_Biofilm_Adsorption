# PINN3D (Catkin Biofilm Adsorption) — Knowledge Transfer

Compiled from a working session reviewing `data.py` / `models.py` / `train.py`
for the Catkin biofilm adsorption paper (companion to the WCF manuscript).
Covers PINN fundamentals, the adsorption chemistry the model encodes, every
bug found in the original code, the fixes applied, and how the loss-spike
investigation was actually resolved.

---

## 1. PINN fundamentals, from a normal neural net upward

**A normal neural network** learns `f(x) → y` by minimizing the gap between
its prediction and real measured data — this is `data_loss`. It has no
concept that `y` should obey any physical law; it only curve-fits whatever
data it's given.

**A physics-informed neural network (PINN)** adds extra loss terms that
penalize the network whenever its output *violates a known equation*,
evaluated at points where there is no real measurement at all. These extra
points are **collocation points** — `data.collocated_input` is a made-up
grid of Time/Concentration/Dosage combinations that exists purely so the
physics loss has somewhere to be evaluated. You're not asking "does this
match data here," you're asking "does this obey the physics here."

**Autograd for physics derivatives.** Checking an equation like PSO kinetics
requires a derivative of the network's output with respect to its input
(`dq/dt`). Because the network is built entirely from differentiable
operations, PyTorch can compute that derivative automatically:

```python
dq_dt_n = torch.autograd.grad(
    outputs=collocation_pred_adsorption.sum(),
    inputs=data.collocated_scaled_input_tensor,
    create_graph=True
)[0]
```

`create_graph=True` matters because this derivative is about to be used
inside *another* loss term that itself gets backpropagated — PyTorch needs
to remember how the derivative was computed (not just its value) so it can
differentiate through it a second time ("double backward").

**Inverse problems — learning physical constants, not just a curve.**
`k2`, `kf`, `n` are `nn.Parameter`s trained by gradient descent, exactly
like the network's weights. This is the PINN "inverse problem" pattern:
find a function *and* a set of physical constants such that the function
obeys the physics defined by those constants — everywhere, not just where
you happened to measure. This is strictly more constrained than a classical
curve fit, which only tunes constants to match one specific dataset with
one specific equation.

**Reparameterizing constants to stay physically valid.** `k2`/`kf` must be
positive; `n` must additionally stay in a bounded range for this system
(see domain section). Rather than constrain the optimizer directly (which
creates hard walls it can get stuck against), the raw parameter is passed
through a smooth transform:

```python
def transform_k2(raw_k2):
    return F.softplus(raw_k2) + K2_FLOOR   # always positive, has a floor

def transform_n(raw_n):
    return N_MIN + (N_MAX - N_MIN) * torch.sigmoid(raw_n)  # bounded both sides
```

The optimizer is free to move the *raw* value anywhere on the real line;
the transform guarantees whatever it picks is automatically physically
valid.

**Why inputs/outputs get standardized (scaled).** Neural nets train better
when inputs/outputs are roughly mean-0, unit-scale — otherwise a network
input like `Time` (0–400) and `Dosage` (0.03–0.12) forces very different
jobs onto the same first-layer weights, slowing and destabilizing training.
`StandardScaler` fixes that. **But this is also exactly where the main bug
came from** (Section 4): once the network lives in scaled space, any
physics equation written in real units (mg/g, minutes) has to explicitly
convert the network's output back before comparing the two — otherwise
you're checking an equation with the two sides in different units.

**Why `predict_qe` iterates instead of solving directly.** The Freundlich
equilibrium + mass balance together give `qe = kf·(Co − qe/dosage)ⁿ`, with
`qe` on both sides. For general (non-integer) `n` there's no closed-form
inverse, so the code uses **damped fixed-point iteration**: guess `qe`,
plug into the right-hand side for a better guess, repeat, but only move
part-way (`damping=0.3`) each step to avoid overshoot/divergence. This is
standard practice for implicit adsorption equations, not something exotic.

---

## 2. Adsorption domain knowledge — what each equation actually says

**Pseudo-second-order kinetics (PSO):**
`dq/dt = k2·(qe − qt)²`

In plain language: *the rate of further dye uptake right now is
proportional to the square of how far current uptake (`qt`) still is from
the equilibrium/maximum uptake (`qe`)*. Far from equilibrium → fast uptake;
near equilibrium → `(qe − qt) → 0` → rate naturally decays to zero. This
means a model correctly satisfying this ODE **cannot** predict adsorption
exceeding `qe`, ever, at any time — that's not a modeling choice, it's
built into the equation. (This is why the *earlier, separate* 1D kinetics
model's long-time extrapolation overshooting its own learned `qe` was a red
flag: a model actually obeying PSO can't do that.)

**Freundlich isotherm:**
`qe = kf·Ce^n`

A snapshot *after* equilibrium: how much gets adsorbed (`qe`) as a function
of how much dye remains in solution at equilibrium (`Ce`). Freundlich
specifically models **heterogeneous surfaces** — real adsorbent binding
sites aren't identical (different pore sizes/surface chemistry), so
adsorption strength varies site to site, unlike Langmuir which assumes one
uniform site energy. The exponent `n` describes how favorable that
heterogeneous adsorption is; sub-linear (`0 < n < 1`) means efficient
adsorption even at low remaining concentration — the usual "favorable"
regime. This is why the code bounds `n` to `[0.05, 1.0]` rather than
leaving it unconstrained: it's a physically-informed prior for this
material class, not an arbitrary choice. (Double-check which convention —
`n` vs. `1/n` — your code uses against how you plan to report it in the
manuscript; conventions vary between papers.)

**Mass balance linking the two:**
`Ce = Co − qe/dosage`

Whatever dye isn't left in solution must be sitting on the adsorbent.
`dosage` (adsorbent-to-solution ratio) converts "amount adsorbed per gram
of material" back into "concentration removed from solution." This
equation is what mechanically ties the kinetics experiment (fixed
concentration, dye disappearing over time) and the isotherm experiment
(fixed time, varying starting concentration) into *one* underlying
physical process, even though they were run as two separate lab
experiments.

**Why unifying kinetics + isotherm in one PINN is the actual novelty.**
Classically: fit PSO to kinetics data alone (one `qe`), fit Freundlich to
isotherm data alone (an independently-fitted `kf`/`n`) — two disconnected
curve fits with no requirement to agree with each other. The 3D PINN here
uses the *same* `kf`/`n` (hence the same physically-derived `qe`) to
explain both datasets simultaneously. That's a real constraint a classical
two-fit approach doesn't have — worth stating explicitly in the paper as
the methodological contribution, not left implicit.

---

## 3. Bugs found in the original code, and the fixes

### 3.1 Scale mismatch between physics losses and the network's output (main bug)

`data_loss` correctly compared the model's *scaled* output against
`scaled_output_tensor`. But `collocation_pred_adsorption` — the same
scaled-space output — was then used **directly**, with no inverse
transform, against real-physical-unit quantities in three places:

- `initial_loss = mean(initial_adsorption_pred**2)` forced the *scaled*
  output to 0 at t=0. That actually forces the *real* adsorption toward
  `output_scaler.mean_`, not toward physical zero uptake at t=0 (the
  correct initial condition for the PSO ODE).
- `pso_loss`: `qe_real` (real mg/g, from `predict_qe`) was compared against
  `collocation_pred_adsorption` (still scaled).
- `iso_loss`: same issue — `qe_real` (real) vs. `max_adsorption_pred`
  (scaled).
- The derivative `dq_dt_real = dq_dt_n * (1/input_stats['Time']['std'])`
  only rescaled the time axis; it was missing the `output_std` factor
  needed to convert a derivative of *scaled* output into a derivative of
  *real* output (chain rule: `d(real)/dt = output_std * d(scaled)/dt`).

**Fix applied:**

```python
output_mean = torch.tensor(data.output_mean, dtype=torch.float32, device=device)
output_std  = torch.tensor(data.output_std,  dtype=torch.float32, device=device)

def unscale_output(scaled_output):
    return scaled_output * output_std + output_mean

# derivative: needs BOTH the time-scale factor and the output-scale factor
dq_dt_real = dq_dt_n * output_std * (1 / data.input_stats['Time']['std'])

collocation_pred_adsorption_real = unscale_output(collocation_pred_adsorption)
pso_loss = torch.mean((dq_dt_real - (positive_k2 * (qe_real - collocation_pred_adsorption_real)**2))**2)

initial_adsorption_pred_real = unscale_output(model(data.initial_scaled_datapoints_tensor))
initial_loss = torch.mean(initial_adsorption_pred_real**2)

max_adsorption_pred_real = unscale_output(model(data.max_time_tensor))
iso_loss = torch.mean((qe_real - max_adsorption_pred_real)**2)
```

`data_loss` needed no change — both sides of that comparison were already
in scaled space.

**Observed effect after the fix:** the data-loss floor dropped roughly two
orders of magnitude (~1e-6 → ~3e-8) — the optimizer could now actually
drive the real data fit down instead of fighting physics losses that were
previously on the wrong scale/offset.

### 3.2 Duplicate `.to(device)` calls that don't reassign

```python
self.collocated_scaled_input_tensor.to(self.config.device)   # no-op: .to() returns a new tensor
```

`.to()` doesn't move a tensor in place — the result has to be reassigned.
This was harmless only because everything runs on CPU and `pass_to_device`
correctly reassigns these tensors later; it would silently fail to move
data the moment GPU is used.

### 3.3 Import collision risk

Two different `data.py` files exist in the project (old single-purpose
`Data` class, new `CombinedData` class). If `src/pinn1D` and `src/pinn3D`
are ever imported in the same Python process (e.g. mixed notebook cells),
a bare `from data import ...` can silently resolve to whichever one loaded
first into `sys.modules`. Fixed with a qualified-import-first pattern:

```python
try:
    from pinn3D.data import CombinedData, load_data
    from pinn3D.models import PINN3DModel, predict_qe
except ImportError:
    from data import CombinedData, load_data
    from models import PINN3DModel, predict_qe
```

(Note: this only actually protects you if `pinn3D`/`pinn1D` are real
Python packages with `__init__.py`; otherwise the qualified import always
fails and it silently falls through to the bare import every time — worth
confirming the package structure is real.)

---

## 4. Loss-spike investigation — how it was actually diagnosed

**Initial hypothesis (before the fix in 3.1):** the recurring spikes in
Initial/ISO loss might be caused by the scale mismatch itself, since
optimizing physical parameters against wrongly-scaled physics targets is a
plausible way to get periodic correction cycles.

**After fixing 3.1:** spikes did not go away — they got more frequent.
This ruled out the scaling bug as the cause of the spikes specifically
(even though fixing it was still correct and necessary).

**Second hypothesis:** the `predict_qe` fixed-point solve (Section 1,
"why `predict_qe` iterates") going unstable as `kf`/`n` drift into a
region where the damped iteration doesn't converge cleanly in 40 steps, or
where the mass-balance term `Ce = co − qe/dosage` goes negative and hits
the `clamp(min=1e-8)` — a hard kink that could feed a limit cycle.

**Instrumentation added to test this:**

```python
def predict_qe(co, dosage, kf, n, n_iter=40, damping=0.3, return_diagnostics=False):
    qe = 0.5 * co * dosage
    clamp_hits = 0
    for _ in range(n_iter):
        Ce_raw = co - qe / dosage
        if return_diagnostics:
            clamp_hits += torch.sum(Ce_raw <= 1e-8).item()
        Ce = torch.clamp(Ce_raw, min=1e-8)
        qe_new = kf * (Ce**n)
        qe = (1 - damping) * qe + damping * qe_new
    if return_diagnostics:
        return qe, {"clamp_hits": clamp_hits, "qe_min": qe.min().item(),
                     "qe_max": qe.max().item(), "qe_mean": qe.mean().item()}
    return qe
```

Logged `K2`, `Kf`, `N`, `Qe_min/max/mean`, `Clamp_hits` every epoch in
`train.py` alongside the four losses.

**Result — hypothesis ruled out.** Across a spike (epochs 3440→3452):
`K2`, `Kf`, `N` moved perfectly smoothly, no jump, no discontinuity.
`Clamp_hits` stayed at 0. `predict_qe` and the physical-parameter
transforms are confirmed clean.

**Actual signature found:** `Data Loss`, `Initial Loss`, and `ISO Loss`
rise and fall together over a ~10-epoch cycle, while `PSO Loss` stays
essentially flat (0.000046–0.000051) the entire time.

**Real cause — unweighted multi-term loss sum interacting with Adam's
adaptive scaling.** `total_loss = data_loss + pso_loss + initial_loss +
iso_loss` sums four terms without weights. Adam divides each parameter's
update by an estimate of that parameter's own gradient variance — so a
loss term with a *small but consistently-directed* gradient (PSO, sitting
near a plateau) can receive an outsized effective step even while its raw
value barely moves. The optimizer nudges the shared network representation
to shave a little more off the PSO residual; because data/initial/max-time
predictions share that same hidden representation, they all move together;
their losses balloon; their now-larger gradients pull the network back;
PSO's pull reasserts; repeat. This is a known PINN failure mode
(cf. "gradient pathology" in physics-informed training, e.g. Wang, Teng &
Perdikaris 2021) — not a bug in `predict_qe`, the scaling fix, or the
parameter transforms. All three of those are now confirmed correct.

**Optional further confirmation (not yet done):** log parameter-gradient
norms `‖∇_θ pso_loss‖` vs. `‖∇_θ data_loss‖` at a calm epoch and a spike
epoch. If PSO's gradient norm stays flat while the others visibly swing,
that directly confirms the Adam-adaptive-scaling mechanism rather than
leaving it as strongly-inferred-but-unconfirmed.

**Fix options, if pursued (not yet implemented), roughly by effort:**
1. **Fixed manual loss weights** — cheapest; use the per-term raw values
   from the logged CSV to set each term to a comparable starting order of
   magnitude.
2. **Gradient-norm-based adaptive weighting** (GradNorm-style) — reweight
   each term periodically by the inverse of its own parameter-gradient
   norm; more robust, more code.
3. **LR decay/scheduler** — doesn't remove the underlying tension, but
   shrinks step size over training, damping the oscillation amplitude as a
   side effect; simplest to add, weakest fix.

Spike amplitude is currently small in absolute terms (total loss peaks
~0.003–0.006), likely not affecting the final converged `k2`/`kf`/`n`
much after 8000 epochs — but it will show up in a loss-curve figure and is
worth fixing rather than explaining away if this is headed for a
higher-IF journal submission.

---

## 5. Concept-to-code mapping

| What was wrong | ML concept | Adsorption concept |
|---|---|---|
| Physics losses compared scaled network output to real `qe_real` | Standardization only helps training — physics equations are unit-aware and don't know about `StandardScaler` | `qe_real` comes from the Freundlich equation in real mg/g; comparing to anything not in mg/g breaks the equation |
| Missing `output_std` in the derivative | Chain rule under a change of variables — scaling the output rescales its derivative too | The derivative *is* `dq/dt` from PSO — must be real mg/g-per-minute to mean anything |
| `initial_loss` forcing scaled output to 0 | Same scaling issue, applied to a boundary/initial condition | `qt(0)=0` — no dye adsorbed before the experiment starts; the PSO ODE's initial condition |
| Suspected (ruled out) fixed-point instability in `predict_qe` | Differentiating through an iterative/implicit solve can be fragile if it doesn't converge cleanly | The implicit equilibrium equation `qe = kf(Co−qe/dosage)ⁿ` has no closed form, hence the iterative solve |
| Actual spike cause: unweighted multi-term loss sum | Adam's per-parameter adaptive scaling can let a small-but-steady gradient (PSO) dominate a step over larger, noisier ones | Not a chemistry issue — purely a training-dynamics artifact of satisfying data fit + initial condition + two coupled physical equations at once |

---

## 6. Where this stands / next steps

- [x] Scale-mismatch bug fixed and confirmed (data loss floor improved ~100x)
- [x] `predict_qe` instrumented; fixed-point instability hypothesis tested and ruled out
- [x] Loss-spike mechanism identified: unweighted multi-loss-term competition under Adam
- [ ] Loss-term reweighting not yet implemented (three options above, undecided)
- [ ] Regenerate fit/extrapolation/residual diagnostic plots for the 3D model specifically (the ones on hand were from the earlier, separate 1D kinetics-only model and don't represent this model)
- [ ] Benchmark against classical PFO/PSO/Freundlich fits with R²/RMSE/AIC for the paper's ML section
- [ ] Decide target journal — Separation and Purification Technology (IF 9.0, Q1) was the best scope/selectivity fit found so far; Journal of Water Process Engineering (IF 6.7) as a fallback; Chemical Engineering Journal (IF 13.2) as a reach

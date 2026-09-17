# PINN3D Paper Assets — Items 1-8

Generated 2026-09-13 from a 4-seed ensemble (seeds 42, 43, 44, 45; reweighted-loss `train.py`, 8000 epochs each) trained in a cloud sandbox on the real dataset (`data of biofilm(TINON BHAI).xlsx`). All figures are in `figures/`.

## Two things to resolve before anything here goes in the manuscript

**1. The classical kinetics/isotherm parameters I computed from your raw Excel data do not match the values recorded from earlier work on this paper.** Fitting PFO/PSO (nonlinear least squares) directly on the `catkin (pfo pso)` sheet gives PSO qe=1.158 mg/g, k2=0.109 — not qe=0.58232 previously noted. That earlier number is roughly half of the actual final measured data point at t=170 (1.17853 mg/g), which is not physically possible for a proper PSO fit (qe must be ≥ the highest observed qt). I checked linearized PSO regression too (qe=1.248) — still doesn't match. I could not reproduce 0.58232 or n=0.846 from any regression method run against the data in this file. This may mean the earlier figures came from a different dataset version, a different material, or a transcription slip somewhere upstream — I can't tell which without your input. **Please check which numbers are the ones actually intended for this manuscript before the benchmark table below is finalized.**

**2. Freundlich exponent convention: the mass-balance-Ce fit gives n>1 (unfavorable/super-linear), not n<1.** Using the same mass-balance Ce that `predict_qe()` uses internally (Ce = Co − qe/dosage), both nonlinear regression (n=1.212) and the log-log linearization (n=1.182, which matches a note in an earlier session almost exactly) put n above 1 — the "unfavorable" regime. `train.py` currently hard-bounds the PINN's N to [0.05, 1.0] (sub-linear/favorable only), based on a comment that "real adsorption data tops out at ~2.5." That bound may be an unvalidated assumption: if the true system is actually in the n>1 regime once you use mass-balance Ce consistently, forcing N∈[0.05,1] could be steering the physical parameter to a biased value rather than letting the data speak. This is worth resolving deliberately — either you have directly-measured Ce values (not mass-balance-derived) that genuinely support n<1, in which case that should be documented as the reason for the bound, or the bound should be revisited.

Everything below is built on the actual code and actual data either way, and is internally consistent; the two flags above are about which numbers are authoritative, not about bugs in what I ran.

---

## 1. Unified 3D response surface — `fig_3d_response_surface.png`

Time × Concentration heatmaps at 5 dosage levels (0.03, 0.0417*, 0.08, 0.1*, 0.12 — *measured), restricted to the experimentally-covered time range (0–170 min). This is the figure that delivers the "performance where characteristics weren't measured" claim: three of the five panels are dosages that were never tested at any time/concentration combination.

**Important limitation found and worth stating in the paper (or fixing first):** I checked whether the network's dosage response tracks its own physics constraint. At the explicit equilibrium anchor (t=400, the only time the `iso_loss` term is enforced), the network matches the analytic Freundlich mass-balance fixed point almost exactly (typically <2% deviation) across the entire dosage range — genuinely good. But at intermediate times away from both the real data (0–170 min) and that t=400 anchor, I found the ensemble predicts a **non-monotonic** dosage response at high concentration (e.g., at t=200, C=60 ppm, predicted capacity rises with dosage up to ~0.105 then declines) — physically implausible, and not something any of the current loss terms directly forbid. I restricted this figure to 0–170 min to stay inside the well-behaved region; dosage extrapolations right at the t=170 edge combined with high dosage (>~0.1) are the least reliable part of the surface. If you want the full 0–400 min surface for the paper, this needs an additional physics anchor (e.g., an equilibrium-consistency constraint at more than one time, not only t=400) before it's trustworthy — I'd treat this as a real to-do, not a footnote.

## 2. Regenerated kinetics/isotherm diagnostic plots — `fig_kinetics_fit_pinn3d.png`, `fig_isotherm_fit_pinn3d.png`

Real data + classical curve + PINN3D ensemble mean ±1 SD + residuals, specific to PINN3D (not the old separate 1D model). PINN3D's residuals are tiny (~1e-3) because these are exactly its training points — see the in-sample caveat below.

## 3. Classical vs. PINN3D benchmark

| | Model | Parameters | R² (in-sample) | RMSE |
|---|---|---|---|---|
| Kinetics | PFO (nonlinear) | qe=1.055, k1=0.0864 | 0.919 | 0.0961 |
| Kinetics | PSO (nonlinear) | qe=1.158, k2=0.1093 | 0.966 | 0.0626 |
| Kinetics | **PINN3D (ensemble)** | K2=0.00465±0.00050 | **0.99998** | 0.00144 |
| Isotherm | Freundlich (mass-balance Ce) | Kf=0.0334, n=1.212 | 0.996 | 0.0373 |
| Isotherm | Langmuir | qmax→1.6×10⁶, KL→0 (degenerate) | 0.973 | 0.1015 |
| Isotherm | **PINN3D (ensemble)** | Kf=0.386±0.024, N=0.426±0.010 | **1.00000** | 0.00126 |

**Honest caveat, stated plainly so it doesn't get oversold in the manuscript:** PINN3D's R² here is against the same points it was trained on (data_loss directly fits them), same as the classical fits are evaluated in-sample too — so this is a fair like-for-like comparison, but PINN3D has vastly more free parameters than the 2-parameter classical models, so a much tighter in-sample fit is expected and is not by itself evidence of better generalization. I did not compute AIC for PINN3D against the classical models — naive parameter-count AIC would be meaningless here (thousands of NN weights vs. 2), and none of the ~30 adsorption-ML papers I checked in the last conversation did that comparison either; they all reported R²/RMSE only for the ML side, which is what I've done here. The GPR/LOO-CV work you asked about next is exactly what would give a genuine held-out generalization number.

## 4. Uncertainty quantification — `fig_uncertainty_calibration.png`

4-seed ensemble standard deviation, sliced along Concentration (at t=170, dosage=0.1) and Time (at C=40, dosage=0.0417). SD is low (~0.002-0.003 mg/g) exactly at the measured points in both slices, and grows sharply just outside the measured range (e.g. ~20-30x higher at C=65 ppm or t=400 min than at the nearest measured point). This is a clean, real, well-calibrated result — the ensemble "knows" where it's extrapolating in concentration and time. Note this calibration was NOT clean for dosage specifically once time is unconstrained (see the #1 caveat above) — the good calibration shown here is for the concentration/time dimensions.

## 5. Dosage analysis — `fig_dosage_tradeoff_C20/40/60.png`

I originally planned a "diminishing returns" removal-% plot, but the actual model prediction doesn't have an interior optimum in removal % — **removal efficiency (%) decreases monotonically as dosage increases** across the whole predicted range (e.g. at C0=40 ppm, t=400 min: 79.5% at dosage=0.03 down to 32.7% at dosage=0.12), while absolute capacity qt keeps rising but with clearly diminishing marginal returns (a genuine interior "elbow" around dosage≈0.10-0.095 for C0=40/60; no clean elbow appears for C0=20 within the scanned range). I verified this against the model's own analytic physics target (`predict_qe`) and it's self-consistent, not a training artifact. **This is entirely a model extrapolation** — dosage was never varied continuously in the real experiments (only 0.0417 and 0.1 were tested), so treat this as a testable prediction for the paper to state as such, not as an experimentally confirmed trend.

## 6. Parameter comparison table + Freundlich convention

See the flagged discrepancies at the top. Numbers as computed:

| Parameter | Classical (this data, nonlinear) | PINN3D (ensemble mean ± SD) |
|---|---|---|
| K2 (PSO rate constant) | 0.1093 | 0.00465 ± 0.00050 |
| Kf (Freundlich constant) | 0.0334 (0.0368 via log-log) | 0.386 ± 0.024 |
| n (Freundlich exponent) | 1.212 (1.182 via log-log) | 0.426 ± 0.010 (architecturally bounded to [0.05,1.0]) |

K2 and Kf/n differ by large factors between classical and PINN3D — expected in part because PINN3D fits kinetics+isotherm jointly with one shared Ce definition (mass balance) while the classical fits here are independent per-dataset regressions, but the magnitude of difference (23x for K2) is large enough to warrant a sentence in the paper explaining why, not just presenting the table.

## 7. Loss curves — `fig_loss_curves_before_after.png`

Before/after the fixed-loss-weight fix, log scale, all four raw loss terms. Visually shows the spike reduction already quantified in the earlier loss-reweighting report — useful supplementary figure, optional for the main text.

## 8. Conclusion section — draft

See `conclusion_draft.md` (separate file) for a full draft paragraph, written from what's actually established so far (characterization, classical kinetics/isotherm results, the joint PINN3D approach, and the acknowledged open items above) rather than overclaiming what isn't yet nailed down.

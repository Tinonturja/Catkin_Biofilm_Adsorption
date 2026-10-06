# Design-density simulation study: pre-registered protocol (v1)

Written before any simulation result was computed. The only quantities inspected beforehand were (a) fits of the six candidate laws to the real kinetics sheet (to set truth parameters) and (b) runtimes. Nothing in this protocol may be changed after results are seen; any necessary change is issued as a numbered amendment with a reason.

This is a **sensitivity analysis under stated assumptions**. The real measurement-noise distribution is unknown (no replicates), and synthetic data are never presented as measurements.

## 1. Question

How do the number and placement of kinetic observations, and the noise level, affect (i) recovery of the data-generating kinetic law, (ii) practical identifiability of its parameters, and (iii) whether flexible learners (tiny neural residual, GP residual, PINN) provide predictive information beyond a classical law selected without knowledge of the truth?

## 2. Data-generating mechanisms (truths)

Fitted to the real kinetics sheet (`catkin (pfo pso)`, 8 points with t > 0); values and log-scale standard errors from `results/design_density_truth_fits.json`:

| Truth | Law | Point estimate | Why included |
|---|---|---|---|
| T1 | PSO | qe 1.158, k2 0.1093 | standard single-phase law |
| T2 | PFO | qe 1.055, k1 0.0864 | standard single-phase law |
| T3 | Elovich | alpha 1.158, beta 6.131 | heterogeneous-surface law with unbounded slow rise |
| T4 | two-site PFO | q1 0.748, ka 0.251, q2 0.786, kb 0.00497 | physically bounded two-rate law (fast + slow site) |
| T5 | PSO + k_id sqrt(t) | qe 0.621, k2 3.69, k_id 0.0441 | Weber-Morris-type additive diffusion term (best real-data AICc) |

Two-site PSO is a candidate but not a truth (it is a near-duplicate of T4's structure).

**Parameter uncertainty.** In every replicate the truth parameters are drawn as log(theta) ~ Normal(log(theta_hat), diag(s^2)), s = min(real-data log-SE, 0.5). A draw is rejected (and redrawn, max 100 tries) if q_true(170) is outside [0.8, 1.6] mg/g.

## 3. Noise scenarios

| ID | Model | Basis |
|---|---|---|
| N1 | additive Gaussian, sd 0.01 mg/g | optimistic lower bound |
| N2 | proportional Gaussian, CV 3 % | magnitude-dependent alternative |
| N3 | additive Gaussian, sd 0.03 mg/g | the residual scale of the best-fitting real-data laws (RMSE 0.020 to 0.036) |

Negative noisy values are not clipped.

## 4. Sampling designs (all inside the observed window 0 < t <= 170 min)

| ID | Rule |
|---|---|
| A_current | the real times 10, 20, 30, 50, 80, 110, 140, 170 (n = 8 only) |
| B_uniform | t_i = 170 i / n |
| C_early | geometric spacing from 2 to 170 min |
| D_late | t_i = 170 sqrt(i / n) |

Densities n in {4, 6, 8, 12, 16, 24, 40}: 4 is the smallest n at which a 2-parameter AICc exists; 40 is five times the real design. Design E (adaptive / optimal) is **excluded** because it requires assuming the truth or a sequential experiment.

## 5. Candidate models and selection

PSO, PFO, Elovich, PSO+sqrt, two-site PSO, two-site PFO; nonlinear least squares on log-parameters, two fixed starts per model (the same starts for every replicate; truths are never used as starts). Selection by minimum AICc among models with n - k - 1 >= 1.

## 6. Stage 1 (classical; 500 replicates per condition)

Conditions: N3 x 5 truths x (3 placements B, C, D x 7 densities + A) = 110; N1 and N2 x 5 truths x (B x 7 densities + A) = 80. Total 190 conditions x 500 = 95,000 replicates. Monte-Carlo SE of a proportion <= 0.022.

Per replicate: selected model; exact-model and phase-class correctness (single-phase = PSO, PFO; slow-phase = the other four); Akaike weight of the truth; for the truth model fitted to the data (oracle M0): log-ratio error of each parameter, Wald 95 % interval coverage (log scale), identifiable flag (finite log-SE < 0.354, i.e. 95 % interval spans less than a factor of 4); prediction RMSE against the noise-free truth for M0 and for the AICc-selected model (M1) on four grids: interpolation 0 to 170 (171 points), early (0, 20], late [120, 170], extrapolation (170, 300] (labelled extrapolation).

Additional stage-1 analyses:
- **Within-fold selection stability** at design A, all noises, 5 truths, 200 replicates: AICc selection repeated in each leave-one-out fold; recorded = fraction of folds agreeing with the full-data choice.
- **Profile-likelihood coverage** for T1 (PSO), design B, n in {8, 40}, N3, 100 replicates: 95 % profile intervals (n ln(SSE/SSEmin) <= 3.84) for qe and k2.

## 7. Stage 2 (learners; 30 replicates per condition)

Conditions: truths T1, T3, T4 (single-phase, unbounded slow, bounded slow) x designs {A (n = 8), B n = 16, B n = 40} x N3 = 9 conditions x 30 replicates = 270. Fewer replicates than stage 1 because one replicate costs ~35 s (vs 0.07 s).

Learners on identical data:
- M0 oracle truth-law fit; M1 AICc-selected classical law.
- M2 PINN, current architecture family (kinetic arm of PINN3D): tanh MLP with LayerNorm, hidden (32, 32) = 1,283 parameters; trainable k2 (softplus + 1e-3) and qe (softplus); losses 180 x data (standardized) + 1 x PSO residual on 100 collocation points in [0, 200] (real units) + 35 x q(0)^2; Adam lr 0.005, 4000 epochs, seed = replicate index. No equilibrium anchor (it needs the isotherm arm).
- M2s small PINN, identical but hidden (8, 8) = 131 parameters (pre-specified fairness check; no other sizes will be tried).
- M3 tiny neural residual on top of M1: Delta(t) = (t/170) MLP(t/170), MLP 1-4-1 tanh (13 weights), weight decay 1e-3, Adam lr 0.01, 2000 epochs, mean of 5 seeds.
- M4 GP residual on top of M1: RBF (length-scale 5 to 500 min by marginal likelihood) + white noise, (0, 0) anchored.

Metrics: the same four prediction grids; for T1, PINN k2 and qe log-ratio errors.

## 8. Decision rules and regions

- Recovery regions for exact-model (and separately class-level) selection probability P: reliable P >= 0.8; transition 0.5 <= P < 0.8; unreliable P < 0.5. Reported per condition; no universal threshold will be claimed.
- A parameter is practically identifiable at a condition if >= 80 % of replicates are identifiable.
- **Learner adds information** at a condition if its interpolation RMSE is lower than M1's with one-sided paired Wilcoxon p < 0.05 **and** the median paired relative improvement is >= 10 %; the same test is also reported for the late and early grids.
- **PINN recovers parameters** (T1 only) if >= 80 % of replicates have |ln(k2_hat/k2)| < ln 1.25 and |ln(qe_hat/qe)| < ln 1.10.

## 9. Hypotheses and pre-stated falsification

| ID | Hypothesis | Falsified if |
|---|---|---|
| H1 | the current design cannot reliably recover the kinetic law | at A, N3, exact P >= 0.8 for at least 4 of 5 truths |
| H2 | density matters | mean over truths (B, N3) of exact P at n = 40 minus n = 8 is < 0.10 |
| H3 | placement matters | at both n = 8 and n = 16 (N3), the spread (max - min) across B, C, D of the mean-over-truths exact P is < 0.10 |
| H4 | flexible residual learners add no information at the current design | M3 or M4 "adds information" (section 8) at A for any truth |
| H5 | the PINN cannot recover PSO parameters at the current design | M2 meets the recovery rule at A for T1 |
| H6 | the PINN becomes useful with denser data | tested, not assumed: supported if at B n = 40 M2 or M2s adds information over M1, or M2 meets the recovery rule |
| H7 | the real-data behaviour is consistent with the simulation | for each truth, P(AICc selects PSO+sqrt at A, N3) is reported; truths with P < 0.05 are called incompatible with the real selection; the simulated within-fold agreement distribution is compared with the real value (1 of 8 folds agreed with the full-data choice) |

## 10. Seeds, stopping, reporting

Master seed 20260929; each condition's generator is seeded with crc32 of its label. Replicate counts are fixed; there are no interim looks. If total runtime exceeds 3 times the estimate, stage-2 replicates may be reduced to 20 by a documented amendment. All conditions are reported, including unfavourable ones.

## Implementation note 1 (before any result was examined)

The first launch showed optimizer overflow warnings. Inspection of the code (not of results) found that a fit ending with a non-finite cost could produce a NaN AICc, and Python's `min()` can then return a NaN-scored model. The run was stopped after ~4 minutes, its partial output was deleted unread, and the code was changed so that fits with non-finite cost or parameters count as failed fits and are excluded from selection. No decision rule, condition, replicate count or seed was changed.

## Implementation note 2 (after the stage-1 output was first read)

The profile-likelihood coverage check returned coverage 0.08 and 0.00 for PSO qe with interval log-widths of 0.008 and 0.000. That is a grid-resolution artefact: the scan step (0.05 in log units) was wider than the interval itself, so only the centre point passed the threshold. The scan was replaced by the union of the original coarse grid and a fine grid (step 0.0025 over +/- 0.3), and only this check was re-run with the same seeds. The check's definition (95 % profile interval, n ln(SSE/SSEmin) <= 3.84) and its conditions are unchanged. No other result is affected.

## Erratum (path only)

Section 2 cites `results/design_density_truth_fits.json`; the file is `results/design_density/truth_fits.json`. Content unchanged.

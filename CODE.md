# CODE — per-script guide

What each script does, what it reads, what it writes, and the decision
that matters. Pipeline order is in README.md; frozen modelling decisions
are in SKILL.md. Headline numbers live in the result JSONs and README;
this file explains the code that produced them.

## 0. download_data.py (added in the reproducibility fix)

Downloads the two benchmark archives from the DOI file base and verifies
them. Standard library only (urllib + hashlib), no extraction: the
archives are RAR v4, so extract the raw-data archive yourself
(`unrar x` or `unar`) and move the 12 `recording_*.mat` files into
`data/raw_data/` at the repo root. Already-downloaded files that pass
size + MD5 checks are skipped, not re-downloaded. `Videos.rar`
(~9.6 GB) is deliberately NOT downloaded — no script in this repo
needs it. Expected sizes and MD5s are in DATA.md (they were verified
against the official .md5 files on 2026-10-24).

## 1. filter_causal.py

Causal per-trajectory filtering. Spec (user instruction, 2026-10-24):
first 11 recordings in chronological order = train, last recording
(22H_58M) = test; each recording holds 6 trajectories separated by
homing dwells, and segmentation cuts at dwell midpoints so every
trajectory is filtered independently (no cross-trajectory filter
state). Position/velocity channels: secondary encoder for joints 1–3,
motor resolver for joints 4–6 (same choice as the official
MASTER_filter_and_combine_datasets.m). Units deg → rad. Filter:
4th-order causal Butterworth, fc = 2 Hz at 250 Hz, per-trajectory
steady-state initialisation (sosfilt with zi); acceleration is the
causal backward difference of filtered velocity at 250 Hz; torque is
filtered the same way; then decimate ×25 to 10 Hz. This is
deliberately NOT the official filtfilt (which is non-causal).

- Inputs: `data/raw_data/recording_*.mat` (12 files).
- Outputs: `filtered_causal.npz` (train 40,015 / test 3,637 samples
  @ 10 Hz, no NaN) + `filtered_causal_meta.json`.
- Note: `filtered_causal.npz` is NOT in the GitHub snapshot (it is a
  regenerable data array); run this script to recreate it.

## 2. phase_calibration.py

Phase calibration, train-only, before any test-set use. Works at
250 Hz from the raw recordings, per trajectory, with the same causal
filter as filter_causal.py. For candidate lag L in {0..5} samples at
250 Hz (= 0..20 ms, past only: qdd(t−L) predicts tau(t), never future
acceleration), it fits a per-row model tau_j = Y_j @ pi_j +
friction_j + hydraulic (joint 2 only) and compares 5-fold CV RMSE
inside training. Result: the curve is FLAT (CV changes < 0.0002 Nm
across all lags), so lags are set to 0 rather than inventing
per-joint lags. The official 4 ms sits inside the same flat region.

- Inputs: `data/raw_data/`, `filtered_causal.npz` (for the split).
- Outputs: `phase_calibration.json`.

## 3. kr300_model.py — library, not a runnable step

KR300 R2500 nominal kinematics from literature DH (Selingue et al.,
JIRS 109 (2023), Table 1, MDH convention
T_i = RotX(α)·TransX(a)·RotZ(θ)·TransZ(d)). Rows (a [m], α [deg],
d [m], θ): 1: 0, 0, 0.675, −θ1; 2: 0.350, −90, 0, θ2−90°… see the
file header for the exact table (joint 3 carries θ3−90° in DH form).
Measurement mapping (meas = KUKA + [0, +90°, −90°, 0, 0, 0]):
θ_DH = [−q1, q2−π/2, q3, −q4, q5, −q6+π], velocity/acceleration sign
vector [−1, +1, +1, −1, +1, −1]. Validated against the KUKA datasheet
(0.675+1.150+1.000 = 2.825 m vs A = 2.826 m; 0.350+1.150+1.000 =
2.500 m vs reach C = 2.496 m) and home FK (upper arm vertical,
forearm horizontal). Provides the pinocchio joint-torque regressor
Y (6×60, 10 barycentric parameters per link) and the RNEA check
(Y@pi = RNEA to 1.4e-14). No DH, geometry, or gear ratios exist in
the dataset or its description PDF, which is why a literature DH was
adopted (user decision, 2026-10-24).

- Gotcha (found and fixed during the run): `meas_to_dh()` expects
  (...,6); passing (6,N) arrays silently mangled the axes. All
  numbers in this repo are post-fix reruns.
- Outputs: none (imported by identify.py, psd_constrained.py,
  residual_analysis.py).

## 4. identify.py

Newton–Euler identification. Model (motor-torque units):
tau_j = r_j · (Y_j(q,qd,qdd) @ pi) + friction_j(qd_j) + hyd_j(q2),
friction = f0 + fv·qd + fc·tanh(20·qd) + fa·tanh(2·qd), hydraulic
(J2 only) = Fourier [sin q2, cos q2, sin 2q2, cos 2q2] in measured q2.
r_1 ≡ 1, so pi is the physical parameter vector divided by the
unknown joint-1 gear ratio u_1, and r_j = u_1/u_j are relative gear
ratios. Alternating bounded least squares. Bounds: mass [0,600],
|mc| ≤ 600, I_diag [0,1000], |I_off| ≤ 500, f0 [−10,10], fv [0,20],
fc [0,20], fa [−20,20], hyd [−40,40], r ∈ [0.2, 5]. Fitted r ≈
[1, 0.958, 1.026, 0.2 (bound), 1.131, 5.0 (bound)] — r4/r6 are weakly
identified and hit bounds; NEVER report them as measured gear
ratios. Only mc/I-origin combinations are identifiable; link masses
alone are not (link-1 pi ≈ 0).

- Inputs: `filtered_causal.npz`, `kr300_model.py`.
- Outputs: `identification.json`, `identification_params.npz`.
- Headline (test): pre-PSD mean NRMSE 0.497 / R² 0.740 — already
  beats the official linear baseline (0.6085 / 0.620).

## 5. psd_constrained.py

PSD-constrained refit — the delivered physics model. Why a refit
rather than a post-hoc projection: post-hoc projection minimises
Frobenius distance in spatial-inertia space, not torque error, and
it degraded joint-1 test R² from 0.898 to 0.425. So this script finds
the torque-optimal parameter vector INSIDE the PSD cone by projected
gradient (FISTA) on the torque loss with the per-link PSD projection
as projector, r held at the pre-PSD values (r4/r6 drift to bounds if
freed), friction/hydraulic refit afterwards with pi fixed.
Pre-projection inertias are genuinely non-PSD (min eigenvalues
L2 −5.43 … L6 −0.90); after projection min ≈ −7e-05.

- Inputs: `identification_params.npz`, `filtered_causal.npz`.
- Outputs: `final_psd.json`, `psd_constrained_params.npz`.
- Naming gap (found by independent reproduction 2026-10-25): §6–§7
  load `final_psd_params.npz`, which no script writes. Before running
  them, copy `psd_constrained_params.npz` → `final_psd_params.npz`
  (see README Pipeline step 4). The original work-directory file
  differs from a fresh run by ≤1.1e-3 (FISTA stopping); headlines
  unaffected at reported precision.
- Headline (test): mean NRMSE 0.568 / R² 0.664. PSD costs mostly
  joint 1 (test R² 0.898 → 0.439); train/test stay close everywhere
  (no obvious overfitting).

## 6. redo_identify_j1.py

Joint-1 inertia correction on top of the PSD model. Diagnosis that
motivated it: post-PSD joint-1 residual correlates +0.899 with qdd1
(test +0.881), slope ≈ 12.0 motor-scaled units, vs pre-PSD +0.278 /
≈1.4 — i.e. the PSD projection itself distorted joint 1's effective
inertia by ~10.6 units. Model: PSD pi frozen, tau_1 += d·qdd_meas,1,
d = 11.957 (train LS). d is deliberately kept OUTSIDE the PSD
projection as an effective/rotor inertia term: putting it inside the
pi fit before projection would let the projection distort it again.
Two variants exist: the post-hoc version (delivered) and a joint
refit of r/friction/hydraulic with d (same d to 4 decimals, same test
mean, but it perturbs the J2 r/hydraulic tradeoff with no test gain,
so it is NOT delivered).

- Inputs: `final_psd_params.npz`, `filtered_causal.npz`.
- Outputs: `j1_correction.json`; delivered parameters
  `j1_correction_posthoc_params.npz` (the joint-refit
  `j1_correction_params.npz` is NOT the delivered model).
- Headline (test): J1 RMSE 2.012 → 0.953 Nm, NRMSE 0.749 → 0.354,
  R² 0.439 → 0.874; mean test NRMSE 0.568 → 0.502, R² 0.664 → 0.737.
  Not full recovery to pre-PSD (J1 0.857 Nm / R² 0.898): PSD also
  distorted a q1-coupled term that a constant d cannot restore.

## 7. residual_analysis.py

Residual analysis for the delivered PSD model (before the J1
correction). Residual = measured tau − predicted tau, on frozen
training parameters; test predictions come from the stored params,
train predictions are recomputed. Candidate correlates (10 Hz):
own q/qd/qdd, |qd|, measured tau, segment index, time in segment,
distance to segment boundary, slow-sample flag (max|qd| < 0.05
rad/s), segment start/end windows (dwell proxies: segments are cut
at dwell midpoints). Findings: slow samples are 17.6% of train
samples but 48.1% of train SSE (test 16.9% / 55.2%); moving vs slow
RMSE (train) J3 0.54 vs 3.36, J5 0.21 vs 1.55, J6 0.20 vs 1.59 Nm.
Residual autocorrelation ~0.64–0.89 at 1 s lag, ~0 at 5–20 s, and
seg-index/global-time corr ≈ 0 — the missing effect lives on a
seconds (dwell/trajectory) timescale, not a day-long drift.

- Inputs: `final_psd_params.npz`, `filtered_causal.npz`.
- Outputs: `residuals.npz`, `residual_analysis.json`, plots
  (PNGs are historical artifacts; new figures are SVG).
- Note: `residuals.npz` is NOT in the GitHub snapshot; run this
  script (after §4–§5) to recreate it.

## 8. residual_plots_corrected.py

Plots and summary stats for the corrected model (PSD + J1 d).
Inputs per its header: `residuals_corrected.npz` +
`filtered_causal.npz`. After the J1 motion error is gone, slow
samples dominate even more: 73.6% of train SSE (was 48.1%) and
78.3% of test SSE (was 55.2%). Corrected-model headline (test):
mean RMSE 1.029 Nm / NRMSE 0.502 / R² 0.736. J1 dwell detail: dwell
-window mean residuals are bimodal ±~1.9 Nm (sign-flipping, so mean
plots cancel to ~0 while RMSE stays large — test slow 1.79 vs
moving 0.66 Nm); magnitude matches fitted J1 Coulomb fc ≈ 1.77 Nm
(static-friction band edge). Last-motion qd1 sign alone does NOT
predict the dwell-offset sign (train agreement 50%).

- Outputs: `residual_analysis_corrected.json`, plots
  (`residual_corrected_*.png`, historical; the J1 dwell-window plot
  is the exception noted in the session record).
- Note: `residuals_corrected.npz` is NOT in the GitHub snapshot;
  see DATA.md for regeneration.

## 9. dwell_diagnosis.py — raw 250 Hz ONLY

Dwell diagnosis (user chose diagnosis over history-friction modelling
and dwell-downweighting). The filtered 10 Hz file has no q_ref /
tau_fb, so this script reads the raw recordings. Dwell = contiguous
run with max|qd_ref| < 0.1 deg/s AND max|q_ref| < 0.01 deg for > 1 s
(84 dwell groups). RULED OUT as the dwell-torque cause: tracking
error (group e std ≤ 0.0046 deg, pooled |r| ≤ 0.12, OLS R² ≤ 0.014);
feedback/feedforward (in dwell tau_ff std ≈ 0.001 Nm, tau_fb ≈ 0
while tau drifts Nm-level); raw tau_meas ≠ tau_ref_ff + tau_fb_meas
(RMS diff 3–10 Nm per joint — never reconstruct total torque from
those channels); gearbox deflection (J3 deflection changes only
0.006 deg while tau moves 7 Nm, corr 0.145). Positive signals:
within-dwell time drift, strongest J3 (bin means −8.5 → −15.5 Nm
over 5.5 s, median slope −1.41 Nm/s, pooled corr −0.634) and J2
(−10.7 → −12.8 Nm, −0.43 Nm/s). Approach-direction group
differences 1.8–3.7 Nm for balanced joints 1/4/5/6, but J2/J3
approach samples are severely unbalanced (J3: 2 vs 70 groups) and
not credible alone.

- Inputs: `data/raw_data/` (raw 250 Hz recordings).
- Outputs: `dwell_diagnosis.json` (included in the repo).

## 10. correlation_expanded.py

Expanded correlations for the corrected model (~140 candidates,
Pearson + Spearman, train primary + test check, moving/slow
stratified). Beyond §7 it adds cross-joint q/qd/qdd, sin/cos
harmonics, Coriolis-type products qd_j·qd_k and centrifugal qd_k²,
causal history (past-window mean qd 0.5/2/5 s, past only), dwell age,
time since segment start / segment position, slow-run approach sign,
and segment-level aggregates. Strongest structure is in MOVING
samples only: a delta-inertia matrix (multivariate resid ~ qdd,
stable train→test: J2 qdd2 −3.6/−3.56, qdd3 −2.25/−2.32; J3 qdd2
−2.49/−2.87, qdd3 +1.19/+1.42; J5 qdd5 +0.51/+0.64; J6 qdd6
+0.42/+0.34; J4 multivariate unstable across sets — do not trust,
univariate qdd4 slope stable +0.64/+0.70). J1 Coriolis-type term:
resid ~ qd1·qd2 Pearson +0.318/+0.289, slope +13.0/+16.5, survives
controlling for qdd, and its slope depends on q2 (low-q2 half
+6.85/+9.06 vs high-q2 +17.42/+21.97) — a pose-dependent coupling,
not a constant coefficient. History signature with regime-dependent
sign: slow samples past-5 s mean qd corr +0.27…+0.43 (J1/J3/J4/J5/
J6, J2 ≈ 0); moving samples past-2 s mean qd corr −0.17…−0.40
(J3–J6). Approach groups (segment-local sign, slow samples): J1
+0.44 vs −1.03, J2 −1.41 vs −0.92, J5 +1.02 vs −0.45, J6 +0.77 vs
−0.89 Nm; J3 has NO negative-approach slow samples (3326 vs 0) —
its direction effect is unidentifiable from this dataset.
Segment-level correlations all |r| < 0.27 (no segment-identity
effect). Dwell-age bins at 10 Hz filtered resolution are mixed and
must NOT be used as dwell-time evidence (clean evidence = raw
250 Hz result in §9).

- Inputs: corrected-model residuals + `filtered_causal.npz`.
- Outputs: `correlation_expanded.json`, 3 plots.

## 11. stage1_motion.py — frozen stage 1

Explicit motion branch, fit ONLY on motion samples (max|qd| ≥ 0.05
rad/s); slow samples are never used in stage-1 fitting. Base:
PSD-model predictions; target e⁽⁰⁾ = tau_meas − tau_phy^PSD (from
`residuals.npz`). Explicit terms (motion-only LS): row 1 ΔM11 =
11.885 + h(q2)·qd1·qd2 with h(q2) = 0.10 + 32.07·sin q2 +
9.73·cos q2 (a/b/c individually ill-conditioned over the narrow q2
range — interpret h as a whole); rows 2/3 block [−3.723, −2.265] /
[−2.499, 1.197] (symmetric shared −2.308 gives identical metrics);
row 4 0.636 (qdd4 only — user: only the stable qdd4 term); row 5
0.505; row 6 0.427. Then r_fast: ridge (α = 10000) on 50 features,
fitted AFTER the explicit terms on the remaining motion residual;
delivered iff recording-block leave-one-out CV on train motion
samples improves ≥ 2% (pre-registered, test untouched) — observed
0.5661 → 0.5305 (+6.3%), delivered. GATING matters: ungated fast
worsened train J2 slow RMSE 3.39 → 3.73 Nm (sin/cos features do not
vanish at slow samples), so delivered r_motion = r_explicit +
motion_flag·r_fast; explicit terms stay ungated (self-gating via
qdd/qd products).

- Inputs: `residuals.npz`, `filtered_causal.npz`.
- Outputs: `stage1_motion.json`, `stage1_motion_params.npz`,
  `stage1_e1.npz` (e⁽¹⁾, input to stage 2).
- Headline (test): motion mean RMSE 0.746 → 0.417 Nm, slow
  unchanged 2.157, overall 1.205 → 0.991. Stage 1 is FROZEN: stage 2
  must not refit it.

## 12. stage2_slow.py — frozen stage 2 (NEGATIVE result)

Causal temporal slow network on top of frozen stage 1. Model:
tau_hat = tau_hat⁽¹⁾ + S·r_slow(z_t), S = diag(0,1,1,0,1,1)
(joints 2,3,5,6 only); z_{t+1} = GRU(z_t, [q, qd, qdd, slow flag,
dwell age capped 10 s /10, per-joint last-motion direction]) =
26 dims, hidden 32. State updates on EVERY sample, loss ONLY on
slow samples, output gated to zero on motion samples, state reset
per segment, causal only, never trained on random samples. Seeds
fixed: torch.manual_seed(0), np.random.seed(0). Early stopping on
the last train recording only; test is evaluation-only. RESULT:
the slow GRU never beat stage 1 on validation (best val slow RMSE
2.868 at epoch 0 vs stage-1-only val 2.637); test slow 2.157 →
2.190, overall 0.991 → 1.005, motion unchanged 0.417 by design.
Train slow fits a little (2.045 → 2.026) and transfers zero.
Generic end-to-end GRU control (same net on e⁽⁰⁾, all joints, all
samples, no gating, no freeze): test motion 0.733 (vs stage-1
0.417), slow 2.176 — it does not recover the explicit motion gains.
Interpretation (frozen): the slow residual is not learnable from
available causal inputs with this GRU config — consistent with a
per-dwell unobserved state (friction state / temperature / pressure
absent from the data). This supports the proposed dwell experiment
(SKILL.md §4): repeated dwells at one fixed pose, approached from
+ and −, holds 2/5/10/20 s, logging torque, motor current,
temperature, pressure, motor-side and link-side angles. Torch is
CPU-only; install gotcha: pip's default TMPDIR may be a small tmpfs
— set TMPDIR to a large disk before installing the wheel.

- Inputs: `stage1_e1.npz`, `filtered_causal.npz`.
- Outputs: `stage2_slow.json`, `stage2_models.pt` (NOT in the
  GitHub snapshot — rerun this script to recreate it).

## Regeneration chain (why the GitHub snapshot is still complete)

The GitHub snapshot holds text files only (upload-connection
limitation): scripts, JSONs, docs, SVGs, logs, conversation. The
excluded binaries are all OUTPUTS, never inputs, of the chain
above: parameter `.npz` files, `stage2_models.pt`, PNG figures,
and the data-bearing arrays (`filtered_causal.npz`, `residuals.npz`,
`residuals_corrected.npz`, `stage1_e1.npz`). Run
download_data.py → the README pipeline order (§1–§12) and every
excluded file regenerates. The raw dataset itself is the only
external input, and it is public, checksummed (DATA.md), and
downloaded by `download_data.py`.

# KR300 Identification Skill

## Name
kr300-identification

## Description
Identify the dynamics of a KUKA KR300 R2500 ultra SE from the Weigand et
al. (2022) Robot Identification Benchmark (DOI 10.26204/data/5), and extend
the identified physics model with a frozen two-stage correction
(motion-branch first, slow/dwell-branch second). Use this skill whenever
the task touches this benchmark, this robot, or this exact pipeline:
causal filtering, literature-DH Newton-Euler regression, PSD projection,
dwell diagnosis, or the phy → phy+motion → phy+motion+slow chain.

## 1. Dataset — read this before touching any file
- Source: Weigand, Götz, Ulmen, Ruskowski (TU Kaiserslautern, 2022),
  *Dataset and Baseline for an Industrial Robot Identification Benchmark*,
  DOI 10.26204/data/5. Description PDF v2.0 (2023-04-09).
- Files: `Robot_Identification_Benchmark_With_Raw_Data.rar`
  (227,384,738 bytes, MD5 `4a9d3adbbca1c9b68577d6db63d58c58`) and
  `Robot_Identification_Benchmark_Without_Raw_Data.rar`
  (12,717,003 bytes, MD5 `6fffa993f60d65f85100bad053888df2`).
  Verify both MD5s against the official .md5 files before use.
  Videos.rar (~9.6 GB) is NOT needed.
- Raw recordings: 12 files, each (6, 90881) float64 @ 250 Hz, channels
  `q_mot_meas, q_se_meas, qd_mot_meas, qd_se_meas, tau_meas, tau_fb_meas,
  q_ref, qd_ref, tau_ref_ff`, plus `time`, `recording_ok`.
  Angles in deg, torque in Nm. `q_se_meas` exists only for joints 1–3;
  joints 4–6 use `q_mot_meas`. `tau_meas` is MOTOR torque (pre-gearbox).
- Chronology: sort recordings by parsed hour/minute, NOT by string
  (21H_3M sorts after 21H_32M as a string but is earlier in time).
  Order: 20H_29M, 20H_38M, 20H_45M, 21H_3M, 21H_11M, 21H_32M, 21H_41M,
  21H_54M, 22H_10M, 22H_41M, 22H_50M, 22H_58M.
- Split (frozen): first 11 recordings = train (66 trajectories),
  last recording (22H_58M) = test (6 trajectories). This matches the
  official split. Each recording = 6 trajectories of ~60.6 s separated by
  homing dwells (~5.5 s, robot at q_ref = 0).
- Do NOT redistribute the raw data or the filtered data arrays in a
  public repo: ship scripts + result JSONs + model parameters, and
  document the download + MD5 + reproduction steps instead.

## 2. Pipeline order — run in this order, do not skip steps
1. `filter_causal.py` → `filtered_causal.npz` (+ `_meta.json`).
   Per trajectory: deg → rad; 4th-order CAUSAL Butterworth fc = 2 Hz
   @ 250 Hz via sosfilt with steady-state initialisation, each trajectory
   initialised independently; acceleration = causal backward difference
   of filtered velocity @ 250 Hz; torque filtered the same; decimate ×25
   to 10 Hz. Deliberately NOT the official `filtfilt`. Expected sizes:
   train 40,015 samples, test 3,637 @ 10 Hz. Trajectories are cut at
   dwell midpoints (dwell = max|qd_ref| < 0.1 deg/s AND max|q_ref| <
   0.01 deg for > 1 s), so every segment starts and ends inside a dwell.
   q ∈ [−2.25, 2.05] rad, qd ∈ [−1.18, 1.23] rad/s, qdd only ±0.9 rad/s².
2. `phase_calibration.py` → `phase_calibration.json`.
   250 Hz, per trajectory, shift qdd by 0–5 samples (0–20 ms, past only),
   decimate; per-joint ridge LS; folds by trajectory, train only.
   RESULT: CV RMSE changes < 0.0002 Nm across all lags — FLAT. Set lags
   to 0. Do not invent per-joint lags from noise. (The official 4 ms
   lies in the same flat region; physical reason: qdd bandwidth < 2 Hz.)
3. `kr300_model.py` + `identify.py` → `identification.json`,
   `identification_params.npz`. Then `psd_constrained.py` →
   `final_psd.json`, `final_psd_params.npz` (the delivered physics model).
4. `redo_identify_j1.py` → `j1_correction.json`,
   `j1_correction_posthoc_params.npz` (delivered J1 correction; the
   joint-refit file `j1_correction_params.npz` is NOT delivered — see §3).
5. Residual/dwell diagnosis: `residual_analysis.py`,
   `residual_plots_corrected.py`, `dwell_diagnosis.py` (raw 250 Hz only),
   `correlation_expanded.py`. See §4.
6. Frozen two-stage model: `stage1_motion.py` → `stage1_motion.json`,
   `stage1_e1.npz`; then `stage2_slow.py` → `stage2_slow.json`,
   `stage2_models.pt`. See §5.

## 3. Identification model — frozen decisions
- No DH/geometry/gear ratios exist in the dataset or PDF. Use literature
  DH (Selingue et al., JIRS 2023, Table 1, MDH):
  rows (a [m], α [deg], d [m], θ):
  1: 0, 0, 0.675, −θ1; 2: 0.350, −90, 0, θ2; 3: 1.150, 0, 0, θ3−90°;
  4: −0.041, −90, 1.000, −θ4; 5: 0, 90, 0, θ5; 6: 0, −90, 0.240, −θ6+180°.
  Validation: 0.675+1.150+1.000 = 2.825 m vs datasheet A = 2.826 m;
  0.350+1.150+1.000 = 2.500 m vs reach C = 2.496 m; home FK = upper arm
  vertical, forearm horizontal.
- Measurement mapping (PDF Eq. 19: meas = KUKA + [0, +90°, −90°, 0, 0, 0]):
  θ_DH = [−q1, q2−π/2, q3, −q4, q5, −q6+π]; velocity/acceleration sign
  vector [−1, +1, +1, −1, +1, −1].
- Model: τ_M,j = r_j·(Y_j@pi) + friction_j(qd_j) + hydraulic_j (J2 only);
  friction = f0 + fv·qd + fc·tanh(20·qd) + fa·tanh(2·qd);
  hydraulic J2 = Fourier [sin q2, cos q2, sin 2q2, cos 2q2] in measured q2;
  r_1 ≡ 1 fixed, so pi is physical pi / unknown joint-1 gear ratio u_1
  and r_j = u_1/u_j. Fit by alternating bounded least squares.
  Bounds: mass [0,600] kg, |mc| ≤ 600, I_diag [0,1000], |I_off| ≤ 500,
  f0 [−10,10], fv [0,20], fc [0,20], fa [−20,20], hyd [−40,40], r ∈ [0.2,5].
- Only mc and I-origin combinations are identifiable; link masses alone
  are not (link-1 pi ≈ 0). r4/r6 hit bounds (weakly identified) — NEVER
  report them as measured gear ratios.
- PSD projection: pre-projection spatial inertias are genuinely non-PSD
  (min eigenvalues L2 −5.43 … L6 −0.90). Project per-link spatial
  inertias to PSD (final min eigenvalues ≈ −7e-05). PSD costs mostly J1:
  test J1 R² 0.898 → 0.439; final PSD test mean NRMSE 0.568 / R² 0.664
  vs official linear baseline 0.6085 / 0.620 (pre-PSD: 0.497 / 0.740).
- J1 correction (delivered): PSD pi FROZEN, τ_1 += d·qdd_meas,1 with
  d = 11.957 (train LS, motor-scaled units). Keep d OUTSIDE the PSD
  projection — it is an effective/rotor inertia term, not link inertia.
  Delivered test: J1 RMSE 2.012 → 0.953 Nm, R² 0.439 → 0.874; mean test
  NRMSE 0.568 → 0.502, R² 0.664 → 0.737. Not full recovery to pre-PSD
  (test J1 0.857 Nm / R² 0.898): PSD also distorted a q1-coupled term.
  The joint-refit variant (r/friction/hydraulic refit + d) gives the
  same d but perturbs the J2 r/hydraulic tradeoff with no test gain —
  do NOT deliver it.

## 4. Residual and dwell diagnosis — what is established
- Slow samples (max|qd| < 0.05 rad/s) are 17.6% of train samples but
  73.6% of corrected-model train SSE (test 16.9% / 78.3%). Moving-phase
  model is already good (e.g. J5/J6 moving RMSE ~0.2 Nm vs slow ~1.6 Nm).
- RULED OUT as dwell-torque cause: tracking error (group e std ≤ 0.0046
  deg, pooled |r| ≤ 0.12, OLS τ~e+ė R² ≤ 0.014); feedback/feedforward
  (in dwell τ_ff std ≈ 0.001 Nm, τ_fb ≈ 0 while τ drifts Nm-level);
  gearbox deflection (J3 deflection changes 0.006 deg while τ moves
  7 Nm, corr 0.145). WARNING: raw τ_meas ≠ τ_ref_ff + τ_fb_meas
  (RMS diff 3–10 Nm per joint) — never reconstruct total torque from
  those two channels.
- Positive: within-dwell time drift, strongest J3 (raw 250 Hz bin means
  −8.5 → −15.5 Nm over 5.5 s, median slope −1.41 Nm/s, pooled corr
  −0.634) and J2 (−10.7 → −12.8 Nm, −0.43 Nm/s). J2/J3 approach-direction
  samples are severely unbalanced (J3 dwell windows 2 vs 70) — J3
  direction effect is UNIDENTIFIABLE from this dataset (slow samples:
  J3 3326 vs 0). Do not use 10 Hz dwell-age bins as dwell-time evidence
  (segment-start ages are lower bounds); the clean drift evidence is
  the raw 250 Hz result.
- J1 dwell detail: dwell-window mean residuals are bimodal ±~1.9 Nm
  (test 11/12 windows |mean| 1.7–2.1 Nm, signs mixed), so mean-based
  plots cancel to ~0 while RMSE stays large (test slow 1.79 vs moving
  0.66 Nm). Magnitude ≈ fitted J1 Coulomb fc ≈ 1.77 Nm (static-friction
  band edge). Last-motion qd1 sign alone does NOT predict the sign
  (train agreement 50%, corr +0.12).
- Expanded correlations (moving samples only; overall correlations are
  diluted by dwell noise): a ΔM inertia-correction matrix — J2: qdd2
  −3.6/−3.56, qdd3 −2.25/−2.32; J3: qdd2 −2.49/−2.87, qdd3 +1.19/+1.42;
  J5: qdd5 +0.51/+0.64; J6: qdd6 +0.42/+0.34 (train→test). J4
  multivariate coefficients are unstable across sets; its univariate
  qdd4 slope is stable (+0.64/+0.70). J1 Coriolis-type term:
  resid ~ qd1·qd2 Pearson +0.318/+0.289, slope +13.0/+16.5, survives
  controlling for qdd (coeff 12.95), and its slope depends on q2
  (low-q2 half +6.85/+9.06 vs high-q2 half +17.42/+21.97).
  History signature with regime-dependent sign: slow samples, past-5 s
  mean qd correlates +0.27…+0.43 (J1/J3/J4/J5/J6); moving samples,
  past-2 s mean qd correlates −0.17…−0.40 (J3–J6).
- Proposed follow-up experiment (NOT run, needs new robot data):
  repeated dwell at one fixed pose, approached from + and − directions,
  holds 2/5/10/20 s, logging torque, motor current, temperature,
  pressure, motor-side and link-side angles, direction-stratified for
  joints 2/3. Existing data cannot answer it (one hold time ~5.5 s,
  unbalanced directions, no temperature/pressure/current channels).

## 5. Frozen two-stage model — the rules that must not be bent
- Order: phy → phy+motion → phy+motion+slow. NEVER train all residuals
  and the temporal net end-to-end first: the slow net re-absorbs ΔM
  and Coriolis-type errors and attribution is lost.
- Stage 1 (motion branch, frozen after fitting):
  τ̂⁽¹⁾ = τ_phy^PSD + r_motion; fit ONLY on motion samples
  (max|qd| ≥ 0.05 rad/s); slow samples never enter stage-1 fitting.
  Explicit terms (per-row LS): row 1 [qdd1, qd1·qd2, sin(q2)·qd1·qd2,
  cos(q2)·qd1·qd2]; row 2 [qdd2, qdd3]; row 3 [qdd2, qdd3];
  row 4 [qdd4] ONLY; row 5 [qdd5]; row 6 [qdd6].
  Fitted values: ΔM11 = 11.885, h(q2) = 0.10 + 32.07·sin q2 +
  9.73·cos q2 (sin/cos coefficients are correlated over the observed
  q2 range — interpret h as a whole, not coefficient by coefficient);
  J2/J3 block [-3.723,-2.265]/[-2.499,1.197] (symmetric shared −2.308
  gives identical metrics); J4 0.636, J5 0.505, J6 0.427.
  Optional r_fast: ridge on 50 shared features (sin/cos q, qd, |qd|,
  qd_i·qd_j except 1-2, tanh(20qd), tanh(2qd)), fitted AFTER explicit
  terms on remaining motion residual; delivered ONLY if recording-block
  leave-one-out CV on train motion samples improves ≥ 2% over
  explicit-only (observed: 0.5661 → 0.5305 at α = 10000, +6.3%).
  CRITICAL: r_fast must be GATED by the motion flag in the delivered
  model (r_motion = r_explicit + motion_flag·r_fast). Ungated fast
  worsened train J2 slow RMSE 3.39 → 3.73 Nm (sin/cos features do not
  vanish at slow samples). Explicit terms stay ungated (their qdd/qd
  products ~0 at slow samples, so slow RMSE is unchanged).
  Stage-1 test result: motion mean RMSE 0.746 → 0.417 Nm, slow
  unchanged 2.157, overall 1.205 → 0.991. Save e⁽¹⁾ for stage 2.
- Stage 2 (slow branch, stage 1 frozen):
  τ̂ = τ̂⁽¹⁾ + S·r_slow(z_t), S = diag(0,1,1,0,1,1) (joints 2,3,5,6 only;
  J1/J4 get NO slow output by design).
  State z_{t+1} = F_φ(z_t, q_t, q̇_t, q̈_t, m_t, d_t, h_t) with m_t =
  slow flag, d_t = slow duration (capped 10 s), h_t = per-joint last
  significant motion direction (|qd| > 0.1 threshold, held otherwise).
  Implementation rules: state updates on EVERY sample of the continuous
  trajectory; loss ONLY on slow samples
  (L = Σ m_t^slow ‖S(e⁽¹⁾ − r_slow)‖² / Σ m_t^slow); slow-branch output
  gated to zero on motion samples; hidden state reset at each
  trajectory start; test time uses only current-trajectory history
  (no future information). Never train on random samples for stage 2:
  random sampling cannot recover second-scale hidden state — use
  continuous trajectory blocks.
  Validation protocol: early stopping on the LAST train recording
  (segments 60–65) only; test is evaluation only, never used for
  selection.
  KNOWN NEGATIVE RESULT (2026-10-24, GRU hidden 32, Adam lr 1e-3):
  the slow GRU did NOT generalise — best validation slow RMSE 2.868
  (epoch 0) vs stage-1-only validation 2.637; test slow 2.157 → 2.190,
  overall 0.991 → 1.005, motion unchanged 0.417 by design. The generic
  end-to-end GRU control (e⁽⁰⁾, all joints, all samples, no gating)
  also failed: test motion 0.733 (vs stage-1 0.417), slow 2.176.
  Interpretation: explicit motion correction transfers and is not
  recovered by this generic net, but the slow residual is not learnable
  from available causal inputs with this GRU configuration — consistent
  with a per-dwell unobserved state. Next candidates, in order:
  (a) linear history-feature slow model (dwell age, approach sign,
  past-velocity windows, slow-only loss); (b) retuned GRU; (c) accept
  the slow floor and run the dwell experiment in §4.
- Ablation chain (report ALL of these, per regime and per joint,
  train AND test): phy; phy+motion; phy+motion+slow; generic residual
  net; two-stage model. Regimes: overall, motion, slow. Also report J1
  and S-joints (2/3/5/6) separately. Never silently relax a frozen
  criterion — change it openly with a user decision.

## 6. Reporting and house rules
- Every number entering a report must be checked against its source
  result file — never write a number from memory into a deliverable.
- Figures: SVG format for all new figures (user instruction 2026-10-24;
  older PNGs are historical artifacts, not to be redone on request
  unless asked).
- After every long task (multi-step compute: training, identification,
  sweeps), write a log file next to the task's work files
  (`logs/<YYYYMMDD>_<task>.md`) with commands run, key results, output
  files, and failures encountered.
- Honesty discipline: report failures as-is; never present inference
  as verified; never present a fitted coefficient as a measured
  physical quantity when it is only identifiable up to a scaling
  (gear ratios, link masses).

## 7. Gotchas
- Pinocchio comes from pip package `pin` (PyPI `pinocchio` is an
  unrelated nose-test plugin — do not install it). Import name is
  `pinocchio`, distribution name is `pin` (4.1.0 verified: Y@pi = RNEA
  to 1.4e-14; pi ordering per body [m, mc_x, mc_y, mc_z, Ixx, Ixy, Iyy,
  Ixz, Iyz, Izz]).
- `meas_to_dh()` expects (...,6) arrays. Passing (6,N) silently mangles
  joints (found and fixed 2026-10-24: joints 2/3 CV ≈ 2.59/2.48 Nm
  pre-fix). All phase/identification numbers are post-fix reruns.
- Official baseline (for comparison only): inverse test mean NRMSE
  0.6085 / R² 0.620 (train 0.5785 / 0.6577); forward test 0.82517 /
  0.3114. Metrics: NRMSE = RMSE/σ, R² vs mean of measured τ (PDF
  definition, not the MATLAB mean(y_sim) quirk).
- Installing torch CPU: pip's default temp dir may be a small tmpfs
  (/tmp 512M) — a large wheel install can die with ENOSPC. Set TMPDIR
  to a directory on a large disk (e.g. ~/workspace/tmp_pip) and use
  `--index-url https://download.pytorch.org/whl/cpu`.
- System python3 used for most scripts: numpy 1.26.4, scipy 1.11.4,
  matplotlib 3.6.3. Venv (pinocchio + torch): pin 4.1.0, numpy 2.5.4,
  scipy 1.18.1, matplotlib 3.11.2, torch 2.14.1+cpu. See
  requirements.txt in the repo.

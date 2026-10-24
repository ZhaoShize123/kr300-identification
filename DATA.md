# Data — Robot Identification Benchmark

## Identity
- Weigand, Götz, Ulmen, Ruskowski (TU Kaiserslautern, 2022), *Dataset
  and Baseline for an Industrial Robot Identification Benchmark*.
- DOI: https://doi.org/10.26204/data/5
- Description PDF v2.0 (2023-04-09):
  https://kluedo.ub.rptu.de/frontdoor/deliver/index/docId/7284/file/Robot_Identification_Benchmark_Description.pdf
- Benchmark intro page: https://www.nonlinearbenchmark.org/benchmarks/industrial-robot
- File base: https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/

## Files to download
| File | Size (bytes) | MD5 (verify against the official .md5 files) |
|---|---|---|
| `Robot_Identification_Benchmark_With_Raw_Data.rar` | 227,384,738 | `4a9d3adbbca1c9b68577d6db63d58c58` |
| `Robot_Identification_Benchmark_Without_Raw_Data.rar` | 12,717,003 | `6fffa993f60d65f85100bad053888df2` |

`Videos.rar` (~9.6 GB) is NOT needed for any script in this repo.
Both archives are RAR v4; extract the raw-data archive and place the
12 `recording_*.mat` files in `data/raw_data/` at the repo root.

## What the raw data contains
- 12 recordings, all 2021-12-15, each (6, 90881) float64 @ 250 Hz
  (363.5 s): `q_mot_meas, q_se_meas, qd_mot_meas, qd_se_meas,
  tau_meas, tau_fb_meas, q_ref, qd_ref, tau_ref_ff`, plus `time`,
  `recording_ok` = 1. Angles in deg, time in s, torque in Nm.
- `q_se_meas` is populated only for joints 1–3; joints 4–6 use
  `q_mot_meas`. `tau_meas` is motor torque (pre-gearbox).
- Chronological order (sort by parsed time, not by string):
  20H_29M, 20H_38M, 20H_45M, 21H_3M, 21H_11M, 21H_32M, 21H_41M,
  21H_54M, 22H_10M, 22H_41M, 22H_50M, 22H_58M.
- Frozen split: first 11 recordings = train (66 trajectories), last
  recording (22H_58M) = test (6 trajectories).

## Regeneration of the excluded arrays
These files are NOT in the repo (they are derived from the raw data
and are regenerable); run the scripts in README order to recreate
them:
- `filtered_causal.npz` ← `filter_causal.py` (expect train 40,015 /
  test 3,637 samples @ 10 Hz)
- `residuals.npz` ← `residual_analysis.py` (PSD-model residuals)
- `residuals_corrected.npz` ← `residual_plots_corrected.py`
- `stage1_e1.npz` ← `stage1_motion.py` (stage-1 residuals, input to
  stage 2)
- `dwell_diagnosis.json` ← `dwell_diagnosis.py` (raw 250 Hz only; the
  JSON is included in the repo, the script needs `data/raw_data/`)

## Official baseline (for comparison)
Inverse test mean NRMSE 0.6085 / R² 0.620; forward test 0.82517 /
0.3114. Metrics: NRMSE = RMSE/σ, R² vs mean of measured τ (PDF
definition). The official difficulty notes (PDF §5.3) match this
repo's findings: standstill/low-velocity phases are the major
challenge — position output is almost constant while motor torque
changes significantly to overcome static friction.

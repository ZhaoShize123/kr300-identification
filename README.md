# KR300 Identification — Frozen Two-Stage Model

Identification of a KUKA KR300 R2500 ultra SE from the Weigand et al.
(2022) Robot Identification Benchmark (DOI 10.26204/data/5), extended
with a frozen two-stage correction: an explicit motion branch first,
then a causal slow/dwell branch. The physics model, the motion branch,
and the negative slow-branch result are all preserved here with the
scripts, numbers, figures, logs, and decision record that produced them.

The reusable playbook for this pipeline is `SKILL.md` (also installed
as a Muse skill, `kr300-identification`). Start there for the frozen
decisions; this README is the map of this repo.

## Headline results (test set, mean RMSE in Nm)

| Model | Overall | Motion | Slow |
|---|---|---|---|
| Physics, PSD-projected (phy) | 1.205 | 0.746 | 2.159 |
| phy + motion (stage 1, frozen) | 0.991 | 0.417 | 2.157 |
| phy + motion + slow (stage 2 GRU) | 1.005 | 0.417 | 2.190 |
| Generic end-to-end GRU (control) | 1.192 | 0.733 | 2.176 |

Physics identification alone already beats the official linear baseline
(test mean NRMSE 0.568 / R² 0.664 vs 0.6085 / 0.620; pre-PSD 0.497 /
0.740). Stage 1 transfers cleanly to the test set and leaves slow
samples untouched. Stage 2 (GRU, this configuration) does NOT
generalise — see `stage2_slow.json` and `logs/` — and is preserved as a
negative result, not a delivered model. The delivered model is
**PSD physics + frozen stage-1 motion branch**.

## Pipeline (run in this order)

1. Download + verify data — see `DATA.md`. Put the 12 raw
   `recording_*.mat` files in `data/raw_data/`.
2. `filter_causal.py` → `filtered_causal.npz` (causal 2 Hz Butterworth,
   per-trajectory steady-state init, decimate ×25 to 10 Hz).
3. `phase_calibration.py` → `phase_calibration.json` (lags = 0; the
   calibration curve is flat, see SKILL.md §2).
4. `kr300_model.py` + `identify.py` → `identification.json`,
   then `psd_constrained.py` → `final_psd.json` (delivered physics).
5. `redo_identify_j1.py` → `j1_correction.json` (delivered J1 term;
   `j1_correction_params.npz` is the joint-refit variant and is NOT
   the delivered model).
6. Diagnosis: `residual_analysis.py`, `residual_plots_corrected.py`,
   `dwell_diagnosis.py` (raw 250 Hz only), `correlation_expanded.py`.
7. Two-stage: `stage1_motion.py` → `stage1_motion.json`,
   `stage1_e1.npz`; then `stage2_slow.py` → `stage2_slow.json`,
   `stage2_models.pt`.

## Repo portability patches (vs the original work directory)
Only three path constants were changed so the scripts run from a fresh
clone; no computation was altered:
- `filter_causal.py`: `BASE`/`OUT` → script dir / `data/raw_data`.
- `phase_calibration.py`: `BASE` → script dir / `data/raw_data`.
- `dwell_diagnosis.py`: `BASE` → script dir / `data/raw_data`, output
  → script dir / `dwell_diagnosis.json`.
All other scripts already used script-dir-relative paths.

## What is in here, and what is deliberately not
- Included: all 12 pipeline scripts, `kr300_model.py`, result JSONs,
  model-parameter `.npz` files, `stage2_models.pt`, figures (PNGs are
  historical artifacts; all new figures are SVG), `logs/`,
  `conversation/conversation.md`, `SKILL.md`, `DATA.md`.
- Excluded: the raw dataset (`.rar`, `extracted/`, `data/` contents)
  and the regenerable data-bearing arrays (`filtered_causal.npz`,
  `residuals.npz`, `residuals_corrected.npz`, `stage1_e1.npz`) — see
  `DATA.md` for download + MD5 + regeneration. Also excluded: virtual
  environments and `__pycache__`.
- `data/` contains only a tracked `.gitkeep` placeholder; put raw
  `.mat` files in `data/raw_data/` (ignored by git).

## Environment
See `requirements.txt`. Pinocchio comes from pip package `pin`
(`import pinocchio`; PyPI `pinocchio` is an unrelated package — do not
install it). Torch is CPU-only and only needed for `stage2_slow.py`;
if pip's temp dir is a small tmpfs, set TMPDIR to a large disk before
installing the wheel (a 512M /tmp once killed this install with
ENOSPC). Tested interpreters: system python3 (numpy 1.26.4, scipy
1.11.4, matplotlib 3.6.3) and a venv (numpy 2.5.4, scipy 1.18.1,
matplotlib 3.11.2, pin 4.1.0, torch 2.14.1+cpu).

## Open items (not finished work)
- Stage-2 next step is undecided: (a) linear history-feature slow
  model, (b) retuned GRU, or (c) accept the slow floor and run the
  proposed dwell experiment (needs new robot data — see SKILL.md §4).
- No license has been chosen for this repo yet.

## File index by stage
- Filtering: `filter_causal.py`, `filtered_causal_meta.json`
- Phase: `phase_calibration.py`, `phase_calibration.json`
- Identification: `kr300_model.py`, `identify.py`,
  `identification.json`, `identification_params.npz`,
  `psd_constrained.py`, `psd_constrained.json`,
  `psd_constrained_params.npz`, `final_psd.json`,
  `final_psd_params.npz`, `test_fit*.png`
- J1 correction: `redo_identify_j1.py`, `j1_correction.json`,
  `j1_correction_posthoc_params.npz` (delivered),
  `j1_correction_params.npz` (not delivered),
  `test_fit_j1_corrected.png`, `j1_residual_vs_qdd.png`
- Diagnosis: `residual_analysis.py`, `residual_analysis.json`,
  `residual_plots_corrected.py`, `residual_analysis_corrected.json`,
  `dwell_diagnosis.py`, `dwell_diagnosis.json`,
  `correlation_expanded.py`, `correlation_expanded.json`,
  `j1_dwell_window_residuals.png`, `residual_*.png`
- Two-stage: `stage1_motion.py`, `stage1_motion.json`,
  `stage1_motion_params.npz`, `stage1_*.svg`, `stage2_slow.py`,
  `stage2_slow.json`, `stage2_models.pt`, `stage2_*.svg`
- Record: `logs/`, `conversation/conversation.md`, `SKILL.md`,
  `DATA.md`

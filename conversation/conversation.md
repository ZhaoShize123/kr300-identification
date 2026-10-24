# Conversation record — KR300 identification session (2026-10-24)

Curated record of the side-chat session that produced this repo.
Fidelity note: this file records every user instruction, every
decision, every headline number, and the file each result lives in.
It is NOT a verbatim dump of the chat: the chat API truncates long
messages on export, and internal runtime messages are excluded. The
verbatim chat history remains in Muse. User instructions below are
quoted where short; the one long design message (two-stage model) is
reproduced in full structure with its formulas.

Session: 2026-10-24, side chat, working language Simplified Chinese.
Standing user rules applied throughout: short direct Chinese replies;
figures in SVG from 23:01 onward (earlier PNGs kept as historical
artifacts); a log file after every long task; numbers in deliverables
checked against source files, never written from memory; frozen
criteria never silently relaxed.

## Decision chain

### 1. Dataset acquisition and verification
- User supplied the benchmark (DOI 10.26204/data/5, Weigand et al.
  2022, KUKA KR300 R2500 ultra SE) and the task.
- User decision: "不用视频，两个数据包就够了" — Videos.rar (~9.6 GB)
  NOT downloaded.
- Outcome: both archives downloaded, both MD5s verified exactly
  against the official .md5 files, extracted (RAR v4). See DATA.md.

### 2. Pipeline specification (user, governing instruction)
- "前 11 条训练，最后 1 条完整测试" (interpreted as recordings:
  first 11 train, last test, matching the official split).
- "单位统一 deg → rad；严格抗混叠滤波：4th-order causal Butterworth
  fc=2 Hz @250 Hz, each trajectory initialized independently,
  downsample ×25 to 10 Hz（deliberately NOT official filtfilt）；
  六关节相位校准：5-fold CV inside training only, only current+past
  20 ms of acceleration；Newton–Euler 物理参数辨识 with
  mass/inertia/friction bounds；per-link spatial-inertia PSD
  projection；final test with parameters frozen from training + fit
  plots."
- Outcome: `filter_causal.py` (train 40,015 / test 3,637 @ 10 Hz).

### 3. DH blocker and user decision
- Finding: no DH/geometry/gear ratios anywhere in the dataset or PDF.
- User chose (options widget): "用文献 KR300 DH 做 Newton-Euler
  回归器" over a structured basis-function fallback.
- Outcome: Selingue et al. JIRS 2023 Table 1 MDH + measurement mapping
  (see SKILL.md §3), validated against the KUKA datasheet and home FK.
  An initial un-shifted conversion was wrong (vertical joint-2 axis)
  and was caught by FK inspection.

### 4. Phase calibration
- Outcome: CV RMSE flat (< 0.0002 Nm) across 0–20 ms lags → lags set
  to 0, no shift invented. Bug found and fixed during this run:
  `meas_to_dh()` expects (...,6) but (6,N) arrays were passed
  initially; all phase/identification numbers are post-fix reruns.

### 5. Identification and PSD projection
- Outcome: pre-PSD test mean NRMSE 0.497 / R² 0.740; final PSD model
  test 0.568 / 0.664 (still beats official baseline 0.6085 / 0.620);
  PSD costs mostly J1 (R² 0.898 → 0.439). r4/r6 hit bounds — weakly
  identified, never to be reported as measured gear ratios.
  Files: `identify.py`, `psd_constrained.py`, `final_psd.json`.

### 6. Diagnosis choice
- Assistant proposed, in priority order: (1) diagnose dwell torque vs
  tracking error / last-motion direction; (2) history-state friction;
  (3) segmented processing / downweight dwell samples.
- User chose (options widget): (1) diagnosis.
- Outcome: tracking error, feedback/feedforward, and gearbox
  deflection RULED OUT; within-dwell time drift confirmed (J3
  −8.5 → −15.5 Nm over 5.5 s raw 250 Hz); J2/J3 approach directions
  severely unbalanced. Files: `dwell_diagnosis.py/json`.

### 7. Follow-up experiment (user-proposed, NOT run)
- User design: repeated dwell at the same pose, approached from +
  and − directions, holds 2/5/10/20 s, recording torque, motor
  current, temperature, pressure, motor-side and link-side angles,
  balanced direction stratification for joints 2/3, testing drift vs
  temperature, hold time, approach direction.
- Assistant refinements (direction defined by velocity sign in final
  1–2 s before stopping; factorial J2/J3 direction design; randomised
  hold order + washout; pressure is J2-only with J3 as control;
  current is not independent evidence of torque; 32 dwells ≈ 25–35 min
  per pose). Needs new robot data — existing data cannot answer it.

### 8. J1 inertia correction (user-ordered)
- User: "附加的关节 1 惯量修正项，重新进行辨识".
- Outcome: PSD pi frozen + τ_1 += d·qdd_meas,1, d = 11.957, kept
  OUTSIDE the PSD projection. Delivered test J1 RMSE 2.012 → 0.953 Nm,
  R² 0.439 → 0.874. Joint-refit variant NOT delivered (same d, worse
  J2 tradeoff, no test gain). Files: `redo_identify_j1.py`,
  `j1_correction_posthoc_params.npz` (delivered).

### 9. Corrected-model residuals (user: "现在的残差呢")
- Outcome: slow samples now 73.6% of train SSE / 78.3% of test SSE;
  no joint shows a large qdd correlation (max |r| 0.25, J4).
  Files: `residual_plots_corrected.py`, 5 figures.

### 10. J1 low-speed question (user)
- User: "这张图上看上去 j1 在低速段显著大于普通速度段啊，为什么
  残差图里没有显著差距呢？"
- Answer: the gap is real but sign-flipping. Dwell-window mean
  residuals are bimodal ±~1.9 Nm (test 11/12 windows |mean| 1.7–2.1
  Nm, mixed signs), so mean-based plots cancel to ~0 while RMSE stays
  large (test slow 1.79 vs moving 0.66 Nm). Magnitude ≈ fitted J1
  Coulomb fc ≈ 1.77 Nm. Last-motion qd1 sign alone does NOT predict
  the sign (train agreement 50%). File:
  `j1_dwell_window_residuals.png`.

### 11. Expanded correlations (user: "做更多有可能的相关性分析")
- Outcome (~140 candidates, train + test, moving/slow stratified):
  a ΔM inertia-correction matrix in moving samples; a J1
  Coriolis-type term with q2-dependent slope; a history/state
  signature whose sign flips by regime; J3 approach-direction effect
  unidentifiable (slow samples 3326 vs 0). Segment-level effects all
  |r| < 0.27. 10 Hz dwell-age bins are NOT dwell-time evidence.
  Files: `correlation_expanded.py/json`, 3 figures.

### 12. SVG rule (user)
- User: "注意：之后的图都要 svg 格式".
- Outcome: all new figures SVG; earlier PNGs kept, not redone.

### 13. Frozen two-stage design (user, reproduced in structure)
- User: "对，顺序就应该这样做，而且要采用冻结式两阶段训练：
  phy → phy+motion → phy+motion+slow。不要一开始把所有残差和
  时序网络一起端到端训练，否则 slow 网络很容易重新吸收 ΔM 和
  Coriolis 型误差，最后无法证明每个模块到底解决了什么。"
- Stage 1: τ̂⁽¹⁾ = τ_phy^PSD + r_motion, with r_motion,k =
  Σ_j ΔM_kj·q̈_j + δ_k1·h(q2)·q̇1·q̇2 + r_fast,k(q,q̇,q̈).
  Rules: motion samples only; explicit ΔM_eff first; J1 gets the
  h(q2) term; J4 keeps only the stable q̈4 term; freeze the whole
  motion branch after training. Target e⁽⁰⁾ = τ_meas − τ_phy^PSD;
  after stage 1 recompute e⁽¹⁾ = τ_meas − τ̂⁽¹⁾.
- Stage 2: τ̂ = τ̂⁽¹⁾ + S·r_slow(z_t), S = diag(0,1,1,0,1,1)
  (joints 2,3,5,6 only). State z_{t+1} = F_φ(z_t, q_t, q̇_t, q̈_t,
  m_t, d_t, h_t) with m_t = motion/slow mode, d_t = slow duration,
  h_t = last significant motion direction. Key implementation detail:
  the temporal net updates state on the whole continuous trajectory
  but loss is computed only on slow samples,
  L_slow = Σ m_t^slow ‖S(e⁽¹⁾ − r_slow(z_t))‖² / Σ m_t^slow;
  motion segments still feed the net for history, but the slow
  branch output is gated to zero on motion segments; hidden state
  resets at each trajectory start; test time uses only current
  trajectory history, no future information.
- Benefits claimed by the user: (1) motion results separately
  verifiable; (2) slow branch cannot pollute the physics correction;
  (3) a strict ablation chain: phy; phy+motion; phy+motion+slow;
  generic residual net; two-stage model — each reporting motion RMSE,
  slow RMSE, overall RMSE, and J1 vs J2/3/5/6 results.
- Training protocol: keep the existing protocol (full-trajectory
  splits, time/trajectory blocks); slow net must NOT train on random
  samples (they cannot recover second-scale hidden state) — use
  continuous input blocks, allow unlabelled moments to update state,
  compute loss only where torque labels exist.
- Final logic (user): "第一阶段修正物理先验在运动状态下的惯量和
  速度耦合误差；第二阶段在冻结运动修正后，用因果时序状态解释低速
  和 dwell 中的历史依赖残差。"
- Outcome: stage 1 delivered (test motion 0.746 → 0.417 Nm, slow
  unchanged); stage 2 GRU did NOT generalise (test slow 2.157 →
  2.190; best validation slow RMSE 2.868 at epoch 0 vs stage-1-only
  2.637); generic control also failed (test motion 0.733).
  Files: `stage1_motion.py`, `stage2_slow.py`,
  `logs/20261024_two_stage_frozen_model.md`.

### 14. Log-file rule (user)
- User: "之后每跑完一个长任务输出一份日志文件".
- Outcome: rule recorded; first log written for the two-stage run
  (`logs/20261024_two_stage_frozen_model.md`); this packaging task
  gets its own log (`logs/20261024_github_skill_package.md`).

### 15. Skill + GitHub packaging (user, this task)
- User: "形成一个 skill，将所有代码，说明，日志以及对话等文件
  上传至我的 GitHub 仓库".
- Outcome: skill written (`SKILL.md`, installed as
  `kr300-identification`); this repo assembled and verified locally;
  push is PENDING the user's repo target and authorisation (gh CLI
  was not logged in; no repo was named). See README for what is
  excluded (raw data, regenerable data arrays, venv) and why.

## Open decisions at packaging time
- Stage-2 next step: (a) linear history-feature slow model,
  (b) retuned GRU, or (c) accept the slow floor and run the dwell
  experiment. Unpicked.
- Dwell experiment: proposed, not run, needs new robot data.
- GitHub target: RESOLVED 2026-10-24 — private repo
  kr300-identification created via browser (user login), App
  installation authorised for it, first snapshot push commit
  5b3b0c14a9280bcc9df334c70be1d8da1c43e39e (37 text files;
  binaries held back by the connector's text-only limit). Private
  empty repos task1–task5 created the same night.
- No license chosen for this repo.

### 16. Reproducibility fix + second push (user, 2026-10-25)
- User requirement: the repo must support FULL reproduction, and
  must contain the conversation record and detailed code
  explanations.
- Audit (against the pushed snapshot, not memory) found: README
  claimed .npz / stage2_models.pt / PNGs were included, but the
  first snapshot holds text files only; conversation.md stopped at
  "push pending" and missed the creation, authorisation, and push;
  no per-script explanation document existed; requirements pinned
  only `pin`; data download was manual.
- Fix in this commit: README inclusion list corrected (snapshot
  text-only vs local-package binaries, regeneration path stated);
  CODE.md added (per-script purpose/inputs/outputs/decisions);
  this section appended; requirements pinned to the tested venv
  set; download_data.py added (stdlib download + size/MD5 verify;
  extraction manual, RAR v4; Videos.rar excluded as before).
- Second snapshot push: this commit, on main, following
  5b3b0c14a9. Binaries remain off GitHub (same connector limit)
  and remain regenerable outputs of the pipeline.

"""Expanded correlation analysis for the corrected model (PSD + J1 d).

Beyond the earlier candidate set, this adds:
  * cross-joint q/qd/qdd, sin/cos harmonics, Coriolis-type products
    qd_j*qd_k and centrifugal qd_k^2,
  * causal history: past-window mean qd_j (0.5/2/5 s, past only),
    dwell age = time since max|qd| last exceeded 0.05 rad/s (per segment),
    time since segment start / segment position,
  * slow-run approach sign per joint (sign of qd_j at the last sample
    with |qd_j| >= 0.05 before the current slow run),
  * stratified correlations (moving-only, slow-only),
  * Spearman (rank) correlations on train,
  * segment-level correlations (segment mean residual vs segment features).

Caveats baked into the interpretation:
  * N=40015 train, so |r| ~ 0.02 is already "statistically" nonzero;
    only |r| >= 0.10 is called weak, >= 0.20 meaningful.
  * Train correlations with design-matrix variables (friction basis,
    regressor columns) are partly least-squares orthogonality, not proof.
  * corr(resid, tau) is mechanical (resid = tau - pred), not a mechanism.
  * Segment starts mid-dwell, so dwell age at segment start is a lower
    bound (the dwell began before the segment).
"""
import json
import os

import numpy as np
from scipy.stats import rankdata

WORK = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(WORK, 'filtered_causal.npz'))
rc = np.load(os.path.join(WORK, 'residuals_corrected.npz'))


def build_candidates(q, qd, qdd, tau, seg, resid):
    N = q.shape[1]
    C = {}
    slow_all = np.max(np.abs(qd), axis=0) < 0.05
    for j in range(6):
        C[f'q{j+1}'] = q[j]
        C[f'qd{j+1}'] = qd[j]
        C[f'qdd{j+1}'] = qdd[j]
        C[f'abs_qd{j+1}'] = np.abs(qd[j])
        C[f'abs_qdd{j+1}'] = np.abs(qdd[j])
        C[f'sin_q{j+1}'] = np.sin(q[j])
        C[f'cos_q{j+1}'] = np.cos(q[j])
        C[f'sin2_q{j+1}'] = np.sin(2 * q[j])
        C[f'cos2_q{j+1}'] = np.cos(2 * q[j])
        C[f'tanh20_qd{j+1}'] = np.tanh(20 * qd[j])
        C[f'tanh2_qd{j+1}'] = np.tanh(2 * qd[j])
        C[f'tau{j+1}'] = tau[j]
        C[f'pred{j+1}'] = tau[j] - resid[j]
    for j in range(6):
        for k in range(6):
            if k != j:
                C[f'qd{j+1}_x_qd{k+1}'] = qd[j] * qd[k]
        for k in range(6):
            if k != j:
                C[f'qd{k+1}_sq'] = qd[k] ** 2
    # per-segment features: t_in_seg, position, dwell age, past means
    t_in = np.zeros(N)
    pos = np.zeros(N)
    dwell_age = np.zeros(N)
    hist = {w: np.zeros((6, N)) for w in (5, 20, 50)}
    approach_sign = np.zeros((6, N))
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        L = len(idx)
        t_in[idx] = np.arange(L) * 0.1
        pos[idx] = np.arange(L) / L
        # dwell age per segment
        last_move = -10 ** 9
        for ii, gi in enumerate(idx):
            if np.max(np.abs(qd[:, gi])) >= 0.05:
                last_move = ii
            dwell_age[gi] = (ii - last_move) * 0.1 if last_move >= 0 else ii * 0.1
        # past-window mean qd (past only) per joint
        for j in range(6):
            cs = np.concatenate([[0.0], np.cumsum(qd[j, idx])])
            for w in (5, 20, 50):
                # mean of samples [i-w, i) ; for i<w use available past
                vals = np.zeros(L)
                for ii in range(L):
                    a = max(0, ii - w)
                    vals[ii] = (cs[ii] - cs[a]) / max(1, ii - a) if ii > 0 else 0.0
                hist[w][j, idx] = vals
            # approach sign: sign of qd_j at last |qd_j|>=0.05 sample before
            # the current slow run; 0 if currently moving / no history
            last_sign = 0.0
            in_slow = False
            for ii, gi in enumerate(idx):
                if np.abs(qd[j, gi]) >= 0.05:
                    last_sign = np.sign(qd[j, gi])
                    in_slow = False
                else:
                    if not in_slow:
                        in_slow = True
                    approach_sign[j, gi] = last_sign
    C['t_in_seg'] = t_in
    C['seg_position'] = pos
    C['dwell_age'] = dwell_age
    C['slow_flag'] = slow_all.astype(float)
    C['seg_index'] = seg.astype(float)
    C['global_time'] = np.arange(N) * 0.1
    for w in (5, 20, 50):
        for j in range(6):
            C[f'hist_qd{j+1}_{w // 10 if w == 50 else w / 10}s'.replace('.0', '')] = hist[w][j]
            # names: 5->0.5s, 20->2s, 50->5s ; fix key below
    # rename history keys cleanly
    for w, nm in ((5, '0.5s'), (20, '2s'), (50, '5s')):
        for j in range(6):
            old = [k for k in C if k.startswith(f'hist_qd{j+1}_') and
                   (k.endswith('0.5s') or k.endswith('2s') or k.endswith('5s'))]
            # the loop above already created messy keys; rebuild directly:
            pass
    for w, nm in ((5, '0.5s'), (20, '2s'), (50, '5s')):
        for j in range(6):
            C[f'hist_qd{j+1}_{nm}'] = hist[w][j]
    for j in range(6):
        C[f'approach_sign{j+1}'] = approach_sign[j]
    return C, slow_all


def pearson_matrix(Y, C):
    names = list(C.keys())
    X = np.column_stack([C[n] for n in names])
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    denom = np.sqrt(np.sum(Xc ** 2, axis=0))[:, None] * np.sqrt(np.sum(Yc ** 2, axis=0))[None, :]
    R = (Xc.T @ Yc) / denom  # (n_cand, 6)
    return names, R


def spearman_matrix(Y, C):
    names = list(C.keys())
    Xr = np.column_stack([rankdata(C[n]) for n in names])
    Yr = np.column_stack([rankdata(Y[:, j]) for j in range(6)])
    Xc = Xr - Xr.mean(axis=0)
    Yc = Yr - Yr.mean(axis=0)
    denom = np.sqrt(np.sum(Xc ** 2, axis=0))[:, None] * np.sqrt(np.sum(Yc ** 2, axis=0))[None, :]
    return names, (Xc.T @ Yc) / denom


out = {}
for split in ['train', 'test']:
    q, qd, qdd, tau = z[f'{split}_q'], z[f'{split}_qd'], z[f'{split}_qdd'], z[f'{split}_tau']
    seg = z[f'{split}_seg_id']
    resid = rc[f'{split}_resid']
    C, slow = build_candidates(q, qd, qdd, tau, seg, resid)
    names, R = pearson_matrix(resid.T, C)
    entry = {'pearson': {}, 'top': {}}
    for j in range(6):
        entry['pearson'][f'joint{j+1}'] = {names[k]: float(R[k, j]) for k in range(len(names))}
        order = np.argsort(-np.abs(R[:, j]))[:12]
        entry['top'][f'joint{j+1}'] = [(names[k], float(R[k, j])) for k in order]
    if split == 'train':
        names_s, Rs = spearman_matrix(resid.T, C)
        entry['spearman_top'] = {}
        for j in range(6):
            order = np.argsort(-np.abs(Rs[:, j]))[:12]
            entry['spearman_top'][f'joint{j+1}'] = [(names_s[k], float(Rs[k, j])) for k in order]
        # stratified: moving-only / slow-only Pearson for key vars
        keys = ['t_in_seg', 'seg_position', 'dwell_age', 'slow_flag'] + \
               [f'hist_qd{j+1}_{nm}' for j in range(6) for nm in ('0.5s', '2s', '5s')] + \
               [f'approach_sign{j+1}' for j in range(6)]
        strat = {}
        for regime, m in [('moving', ~slow), ('slow', slow)]:
            Cm = {k: C[k][m] for k in keys}
            Rm = pearson_matrix(resid[:, m].T, Cm)[1]
            strat[regime] = {}
            for j in range(6):
                strat[regime][f'joint{j+1}'] = {keys[k]: float(Rm[k, j]) for k in range(len(keys))}
        entry['stratified'] = strat
        # slow-run approach group means (slow samples only)
        grp = {}
        for j in range(6):
            m = slow
            sgn = C[f'approach_sign{j+1}'][m]
            rj = resid[j, m]
            grp[f'joint{j+1}'] = {
                'approach_pos_mean': float(rj[sgn > 0].mean()) if np.any(sgn > 0) else None,
                'approach_neg_mean': float(rj[sgn < 0].mean()) if np.any(sgn < 0) else None,
                'approach_pos_n': int(np.sum(sgn > 0)),
                'approach_neg_n': int(np.sum(sgn < 0)),
            }
        entry['slow_approach_groups'] = grp
        # segment-level: segment mean residual vs segment features
        seg_feats = {}
        seg_means = {}
        for s in np.unique(seg):
            idx = np.where(seg == s)[0]
            seg_means[s] = resid[:, idx].mean(axis=1)
            seg_feats[s] = {
                'mean_abs_qd': float(np.mean(np.max(np.abs(qd[:, idx]), axis=0))),
                'max_qd': float(np.max(np.abs(qd[:, idx]))),
                'q_range': float(np.max(q[:, idx], axis=1).max() - np.min(q[:, idx], axis=1).min()),
                'duration': float(len(idx) * 0.1),
            }
        S = np.array([seg_means[s] for s in sorted(seg_means)])  # (66,6)
        F = {k: np.array([seg_feats[s][k] for s in sorted(seg_feats)]) for k in seg_feats[sorted(seg_feats)[0]]}
        seg_corr = {}
        for j in range(6):
            seg_corr[f'joint{j+1}'] = {k: float(np.corrcoef(S[:, j], v)[0, 1]) for k, v in F.items()}
        entry['segment_level_corr'] = seg_corr
    out[split] = entry
    print(f'== {split} top Pearson per joint ==')
    for j in range(6):
        print(f" J{j+1}: " + '  '.join(f'{n} {v:+.3f}' for n, v in entry['top'][f'joint{j+1}'][:8]))

json.dump(out, open(os.path.join(WORK, 'correlation_expanded.json'), 'w'), indent=1)
print('\nsaved correlation_expanded.json')

# plots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tr = out['train']
# 1. top correlates per joint (train Pearson, top 8, excluding tau/pred and lag artefacts is NOT done: show all, tau marked)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for j in range(6):
    ax = axes.flat[j]
    top = [(n, v) for n, v in tr['top'][f'joint{j+1}'] if n not in (f'tau{j+1}', f'pred{j+1}')][:8]
    top = top[::-1]
    ax.barh([n for n, _ in top], [v for _, v in top])
    ax.axvline(0, color='k', linewidth=0.6)
    ax.set_title(f'joint {j+1}: top Pearson r (train, tau/pred excluded)')
    ax.grid(True, alpha=0.3, axis='x')
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(8)
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'correlation_expanded_top.png'), dpi=150)
plt.close(fig)

# 2. dwell-age binned residual mean (slow samples only, train)
ztr = z['train_q'], z['train_qd'], z['train_qdd'], z['train_tau']
q_tr, qd_tr = ztr[0], ztr[1]
seg_tr = z['train_seg_id']
res_tr = rc['train_resid']
C_tr, slow_tr = build_candidates(q_tr, qd_tr, z['train_qdd'], z['train_tau'], seg_tr, res_tr)
age = C_tr['dwell_age']
fig, ax = plt.subplots(figsize=(10, 5))
bins = np.array([0, 0.5, 1, 2, 3, 4, 5, 7])
for j in [0, 1, 2]:
    means = []
    for b in range(len(bins) - 1):
        m = slow_tr & (age >= bins[b]) & (age < bins[b + 1])
        means.append(res_tr[j, m].mean() if m.sum() > 20 else np.nan)
    ax.plot((bins[:-1] + bins[1:]) / 2, means, marker='o', label=f'joint {j+1}')
ax.axhline(0, color='k', linewidth=0.6)
ax.set_xlabel('dwell age (s, time since last moving sample; segment start is a lower bound)')
ax.set_ylabel('mean residual (Nm), slow samples only')
ax.set_title('Train: residual vs dwell age (slow samples)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'correlation_expanded_dwell_age.png'), dpi=150)
plt.close(fig)

# 3. slow approach-direction group means (train)
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(6)
pos = [tr['slow_approach_groups'][f'joint{j+1}']['approach_pos_mean'] for j in range(6)]
neg = [tr['slow_approach_groups'][f'joint{j+1}']['approach_neg_mean'] for j in range(6)]
ax.bar(x - 0.2, pos, 0.4, label='last-motion qd_j > 0')
ax.bar(x + 0.2, neg, 0.4, label='last-motion qd_j < 0')
ax.axhline(0, color='k', linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f'J{j+1}' for j in range(6)])
ax.set_ylabel('mean residual (Nm), slow samples only')
ax.set_title('Train: slow-sample residual by last-motion direction of the same joint')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'correlation_expanded_approach.png'), dpi=150)
plt.close(fig)
print('saved correlation_expanded_top.png, correlation_expanded_dwell_age.png, correlation_expanded_approach.png')

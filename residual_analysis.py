"""Residual analysis for the delivered PSD model (final_psd_params).

Residual = measured tau - predicted tau, on frozen training parameters.
Test predictions are taken from final_psd_params.npz (stored); train
predictions are recomputed with identify.compute_Y / predict (pinocchio).
Candidate correlates (10 Hz filtered data): own q/qd/qdd, |qd|, measured
tau, segment index, time in segment, distance to segment boundary,
slow-sample flag (max|qd| < 0.05 rad/s across joints), and segment
start/end windows (dwell proxies: segments are cut at dwell midpoints,
so segment start = second half of a dwell, segment end = first half).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identify import compute_Y, predict  # noqa: E402

WORK = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(WORK, 'filtered_causal.npz'))
fp = np.load(os.path.join(WORK, 'final_psd_params.npz'))
theta, r = fp['theta'], fp['r']

out = {}
for split in ['train', 'test']:
    q, qd, qdd, tau = z[f'{split}_q'], z[f'{split}_qd'], z[f'{split}_qdd'], z[f'{split}_tau']
    seg = z[f'{split}_seg_id']
    if split == 'test':
        pred = fp['test_pred']
    else:
        print('computing Y train ...', flush=True)
        Y = compute_Y(q, qd, qdd)
        pred = predict(Y, r, theta, q, qd)
    resid = tau - pred
    out[split] = {'pred': pred, 'resid': resid}
    print(split, 'resid RMSE per joint:',
          np.round(np.sqrt(np.mean(resid ** 2, axis=1)), 3), flush=True)

np.savez_compressed(os.path.join(WORK, 'residuals.npz'),
                    train_pred=out['train']['pred'], train_resid=out['train']['resid'],
                    test_pred=out['test']['pred'], test_resid=out['test']['resid'])

# ---- correlation analysis (train primary, test check) ----
def corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

def seg_features(seg, N):
    t_in_seg = np.zeros(N)
    dist_bound = np.zeros(N)
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        t_in_seg[idx] = np.arange(len(idx)) * 0.1
        dist_bound[idx] = np.minimum(np.arange(len(idx)),
                                     len(idx) - 1 - np.arange(len(idx))) * 0.1
    return t_in_seg, dist_bound

summary = {}
for split in ['train', 'test']:
    q, qd, qdd, tau = z[f'{split}_q'], z[f'{split}_qd'], z[f'{split}_qdd'], z[f'{split}_tau']
    seg = z[f'{split}_seg_id']
    resid = out[split]['resid']
    N = q.shape[1]
    t_in_seg, dist_bound = seg_features(seg, N)
    slow = np.max(np.abs(qd), axis=0) < 0.05
    # segment start/end windows: first/last 25 samples (2.5 s) of each segment
    start_win = np.zeros(N, bool); end_win = np.zeros(N, bool)
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        start_win[idx[:25]] = True; end_win[idx[-25:]] = True
    mid = ~(start_win | end_win)
    cands = {}
    for j in range(6):
        cands[f'joint{j+1}'] = {
            'q_j': corr(resid[j], q[j]), 'qd_j': corr(resid[j], qd[j]),
            'qdd_j': corr(resid[j], qdd[j]), 'abs_qd_j': corr(resid[j], np.abs(qd[j])),
            'tau_j': corr(resid[j], tau[j]),
            'seg_index': corr(resid[j], seg.astype(float)),
            'global_time': corr(resid[j], np.arange(N) * 0.1),
            't_in_seg': corr(resid[j], t_in_seg),
            'dist_to_boundary': corr(resid[j], dist_bound),
            'slow_flag': corr(resid[j], slow.astype(float)),
            'lag1_autocorr': corr(resid[j, :-1], resid[j, 1:]),
        }
    # SSE share and RMSE by regime
    regimes = {}
    for name, m in [('slow', slow), ('moving', ~slow),
                    ('seg_start_2.5s', start_win), ('seg_end_2.5s', end_win),
                    ('seg_middle', mid)]:
        regimes[name] = {
            'frac': float(m.mean()),
            'rmse_per_joint': np.sqrt(np.mean(resid[:, m] ** 2, axis=1)).tolist(),
            'sse_share': float(np.sum(resid[:, m] ** 2) / np.sum(resid ** 2)),
        }
    # mean residual in segment start vs end windows, per joint
    win_means = {'start': resid[:, start_win].mean(axis=1).tolist(),
                 'end': resid[:, end_win].mean(axis=1).tolist(),
                 'middle': resid[:, mid].mean(axis=1).tolist(),
                 'slow': resid[:, slow].mean(axis=1).tolist(),
                 'moving': resid[:, ~slow].mean(axis=1).tolist()}
    summary[split] = {'corr': cands, 'regimes': regimes, 'window_means': win_means}
    print(f"\n== {split} ==")
    for j in range(6):
        c = cands[f'joint{j+1}']
        print(f"J{j+1}: " + '  '.join(f"{k} {v:+.3f}" for k, v in c.items()))
    for name, rg in regimes.items():
        print(f"  {name}: frac {rg['frac']:.3f} SSE share {rg['sse_share']:.3f} "
              f"RMSE {np.round(rg['rmse_per_joint'], 2)}")
    print('  window means start/end/mid:', {k: np.round(v, 2).tolist() for k, v in win_means.items()})

json.dump(summary, open(os.path.join(WORK, 'residual_analysis.json'), 'w'), indent=1)
print('\nsaved residuals.npz, residual_analysis.json')

# ---- plots ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. test residual time series, 6 panels, segment boundaries
q_te, tau_te = z['test_q'], z['test_tau']
seg_te = z['test_seg_id']; resid_te = out['test']['resid']
t = np.arange(q_te.shape[1]) * 0.1
fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)
bounds = [np.where(seg_te == s)[0][0] * 0.1 for s in np.unique(seg_te)[1:]]
for j in range(6):
    ax = axes[j]
    ax.plot(t, resid_te[j], linewidth=0.6)
    ax.axhline(0, color='k', linewidth=0.6)
    for b in bounds:
        ax.axvline(b, color='r', linewidth=0.5, alpha=0.5)
    ax.set_ylabel(f'J{j+1} resid (Nm)')
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('time (s, test recording; red lines = segment boundaries = dwell midpoints)')
fig.suptitle('KR300 PSD model: test-set residuals (measured - predicted)')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'residual_test_timeseries.png'), dpi=150)
plt.close(fig)

# 2. correlation heatmap (train)
cands_names = ['q_j', 'qd_j', 'qdd_j', 'abs_qd_j', 'tau_j', 'seg_index',
               'global_time', 't_in_seg', 'dist_to_boundary', 'slow_flag', 'lag1_autocorr']
M = np.array([[summary['train']['corr'][f'joint{j+1}'][c] for c in cands_names] for j in range(6)])
fig, ax = plt.subplots(figsize=(12, 5))
im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(cands_names))); ax.set_xticklabels(cands_names, rotation=30, ha='right')
ax.set_yticks(range(6)); ax.set_yticklabels([f'joint {j+1}' for j in range(6)])
for i in range(6):
    for k in range(len(cands_names)):
        ax.text(k, i, f'{M[i, k]:+.2f}', ha='center', va='center', fontsize=8)
ax.set_title('Train residuals: Pearson correlation with candidate variables')
fig.colorbar(im, ax=ax, label='r')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'residual_correlation_heatmap.png'), dpi=150)
plt.close(fig)

# 3. residual vs segment position: mean residual in 10 position bins + slow/moving
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
for j in range(6):
    ax = axes.flat[j]
    q, seg = z['train_q'], z['train_seg_id']
    resid = out['train']['resid']
    pos = np.zeros(q.shape[1])
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        pos[idx] = np.arange(len(idx)) / len(idx)
    bins = np.linspace(0, 1, 11)
    bc, bm = [], []
    for b in range(10):
        m = (pos >= bins[b]) & (pos < bins[b + 1])
        bc.append((bins[b] + bins[b + 1]) / 2); bm.append(resid[j, m].mean())
    ax.plot(bc, bm, marker='o', linewidth=1.2)
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_title(f'joint {j+1}')
    ax.set_xlabel('position in segment (0=start/dwell, 1=end/dwell)')
    ax.set_ylabel('mean resid (Nm)')
    ax.grid(True, alpha=0.3)
fig.suptitle('Train residuals: mean by position inside segment (segments cut at dwell midpoints)')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'residual_vs_segment_position.png'), dpi=150)
plt.close(fig)
print('saved residual_test_timeseries.png, residual_correlation_heatmap.png, residual_vs_segment_position.png')

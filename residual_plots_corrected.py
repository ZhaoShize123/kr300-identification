"""Corrected-model residual plots (PSD + J1 inertia correction, d=11.957).

Inputs: work/residuals_corrected.npz + work/filtered_causal.npz.
Outputs (all in work/):
  residual_corrected_timeseries_test.png
  residual_corrected_heatmap.png            (train + test correlation heatmaps)
  residual_corrected_vs_segment_position.png (train, mean by position in segment)
  residual_corrected_slow_vs_moving.png      (per-joint RMSE slow vs moving)
  residual_corrected_vs_speed_dwell.png      (mean resid vs |qd| bins; J2/J3
                                              dwell-window means)
  residual_analysis_corrected.json
"""
import json
import os

import numpy as np

WORK = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(WORK, 'filtered_causal.npz'))
rc = np.load(os.path.join(WORK, 'residuals_corrected.npz'))

CANDS = ['q_j', 'qd_j', 'qdd_j', 'abs_qd_j', 'tau_j', 'seg_index',
         'global_time', 't_in_seg', 'dist_to_boundary', 'slow_flag', 'lag1_autocorr']


def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float('nan')


def seg_features(seg):
    N = len(seg)
    t_in = np.zeros(N)
    dist = np.zeros(N)
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        t_in[idx] = np.arange(len(idx)) * 0.1
        dist[idx] = np.minimum(np.arange(len(idx)), len(idx) - 1 - np.arange(len(idx))) * 0.1
    return t_in, dist


summary = {}
for split in ['train', 'test']:
    q, qd, qdd, tau = z[f'{split}_q'], z[f'{split}_qd'], z[f'{split}_qdd'], z[f'{split}_tau']
    seg = z[f'{split}_seg_id']
    resid = rc[f'{split}_resid']
    N = q.shape[1]
    t_in, dist = seg_features(seg)
    slow = np.max(np.abs(qd), axis=0) < 0.05
    c = {}
    for j in range(6):
        c[f'joint{j+1}'] = {
            'q_j': corr(resid[j], q[j]), 'qd_j': corr(resid[j], qd[j]),
            'qdd_j': corr(resid[j], qdd[j]), 'abs_qd_j': corr(resid[j], np.abs(qd[j])),
            'tau_j': corr(resid[j], tau[j]),
            'seg_index': corr(resid[j], seg.astype(float)),
            'global_time': corr(resid[j], np.arange(N) * 0.1),
            't_in_seg': corr(resid[j], t_in),
            'dist_to_boundary': corr(resid[j], dist),
            'slow_flag': corr(resid[j], slow.astype(float)),
            'lag1_autocorr': corr(resid[j, :-1], resid[j, 1:]),
        }
    summary[split] = c

json.dump(summary, open(os.path.join(WORK, 'residual_analysis_corrected.json'), 'w'), indent=1)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. test residual timeseries
seg_te = z['test_seg_id']
res_te = rc['test_resid']
t = np.arange(res_te.shape[1]) * 0.1
bounds = [np.where(seg_te == s)[0][0] * 0.1 for s in np.unique(seg_te)[1:]]
fig, axes = plt.subplots(6, 1, figsize=(11, 12), sharex=True)
for j in range(6):
    ax = axes[j]
    ax.plot(t, res_te[j], linewidth=0.6)
    ax.axhline(0, color='k', linewidth=0.6)
    for b in bounds:
        ax.axvline(b, color='r', linewidth=0.5, alpha=0.5)
    ax.set_ylabel(f'J{j+1} resid (Nm)')
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('time (s, test recording; red lines = segment boundaries = dwell midpoints)')
fig.suptitle('Corrected model (PSD + J1 correction): test-set residuals')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'residual_corrected_timeseries_test.png'), dpi=150)
plt.close(fig)

# 2. correlation heatmaps, train and test
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
for ax, split in zip(axes, ['train', 'test']):
    M = np.array([[summary[split][f'joint{j+1}'][c] for c in CANDS] for j in range(6)])
    im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(CANDS)))
    ax.set_xticklabels(CANDS, rotation=30, ha='right')
    ax.set_yticks(range(6))
    ax.set_yticklabels([f'joint {j+1}' for j in range(6)])
    for i in range(6):
        for k in range(len(CANDS)):
            ax.text(k, i, f'{M[i, k]:+.2f}', ha='center', va='center', fontsize=8)
    ax.set_title(f'{split} residuals: Pearson r')
fig.colorbar(im, ax=axes, label='r')
fig.suptitle('Corrected-model residuals: correlation with candidate variables')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'residual_corrected_heatmap.png'), dpi=150)
plt.close(fig)

# 3. mean residual by position in segment (train)
q_tr, seg_tr = z['train_q'], z['train_seg_id']
res_tr = rc['train_resid']
pos = np.zeros(q_tr.shape[1])
for s in np.unique(seg_tr):
    idx = np.where(seg_tr == s)[0]
    pos[idx] = np.arange(len(idx)) / len(idx)
bins = np.linspace(0, 1, 11)
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for j in range(6):
    ax = axes.flat[j]
    bm = []
    for b in range(10):
        m = (pos >= bins[b]) & (pos < bins[b + 1])
        bm.append(res_tr[j, m].mean())
    ax.plot((bins[:-1] + bins[1:]) / 2, bm, marker='o', linewidth=1.2)
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_title(f'joint {j+1}')
    ax.set_xlabel('position in segment (0=start/dwell, 1=end/dwell)')
    ax.set_ylabel('mean resid (Nm)')
    ax.grid(True, alpha=0.3)
fig.suptitle('Corrected model, train: mean residual by position inside segment')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'residual_corrected_vs_segment_position.png'), dpi=150)
plt.close(fig)

# 4. slow vs moving RMSE bars, train and test
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, split in zip(axes, ['train', 'test']):
    qd = z[f'{split}_qd']
    resid = rc[f'{split}_resid']
    slow = np.max(np.abs(qd), axis=0) < 0.05
    x = np.arange(6)
    slow_rmse = np.sqrt(np.mean(resid[:, slow] ** 2, axis=1))
    mov_rmse = np.sqrt(np.mean(resid[:, ~slow] ** 2, axis=1))
    ax.bar(x - 0.2, slow_rmse, 0.4, label='slow (max|qd|<0.05)')
    ax.bar(x + 0.2, mov_rmse, 0.4, label='moving')
    ax.set_xticks(x)
    ax.set_xticklabels([f'J{j+1}' for j in range(6)])
    ax.set_ylabel('residual RMSE (Nm)')
    ax.set_title(f'{split}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('Corrected model: residual RMSE, slow vs moving samples')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'residual_corrected_slow_vs_moving.png'), dpi=150)
plt.close(fig)

# 5. mean residual vs own |qd| bins (train) + J2/J3 dwell-window means
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
qd_tr = z['train_qd']
bins_qd = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.3])
for j in [1, 2, 4]:
    means = []
    for b in range(len(bins_qd) - 1):
        m = (np.abs(qd_tr[j]) >= bins_qd[b]) & (np.abs(qd_tr[j]) < bins_qd[b + 1])
        means.append(res_tr[j, m].mean() if m.sum() else np.nan)
    ax.plot((bins_qd[:-1] + bins_qd[1:]) / 2, means, marker='o', label=f'joint {j+1}')
ax.axhline(0, color='k', linewidth=0.6)
ax.set_xlabel('own |qd| (rad/s, bin centre)')
ax.set_ylabel('mean resid (Nm)')
ax.set_title('Train: mean residual vs speed')
ax.legend()
ax.grid(True, alpha=0.3)
ax = axes[1]
start = np.zeros(len(seg_tr), bool)
end = np.zeros(len(seg_tr), bool)
for s in np.unique(seg_tr):
    idx = np.where(seg_tr == s)[0]
    start[idx[:25]] = True
    end[idx[-25:]] = True
mid = ~(start | end)
x = np.arange(6)
for name, m, off in [('seg start (late dwell)', start, -0.25), ('middle', mid, 0.0),
                     ('seg end (early dwell)', end, 0.25)]:
    ax.bar(x + off, res_tr[:, m].mean(axis=1), 0.25, label=name)
ax.axhline(0, color='k', linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f'J{j+1}' for j in range(6)])
ax.set_ylabel('mean resid (Nm)')
ax.set_title('Train: mean residual by segment window')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('Corrected model, train: speed dependence and dwell-window means')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'residual_corrected_vs_speed_dwell.png'), dpi=150)
plt.close(fig)
print('saved 5 corrected residual plots + residual_analysis_corrected.json')

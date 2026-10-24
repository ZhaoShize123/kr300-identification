"""Stage 1 of the frozen two-stage model: motion branch only.

Base (phy): PSD model predictions. Target e^(0) = tau_meas - tau_phy^PSD,
taken from work/residuals.npz (PSD residuals, train recomputed / test stored).
Motion samples only: max_j |qd_j| >= 0.05 rad/s. Slow samples are NEVER
used in stage-1 fitting.

Explicit motion branch (per row, least squares on motion samples):
  row 1: [qdd1, qd1*qd2, sin(q2)*qd1*qd2, cos(q2)*qd1*qd2]
         -> [dM11, a, b, c], i.e. dM11 + h(q2)*qd1*qd2 with
         h(q2) = a + b*sin(q2) + c*cos(q2)
  row 2: [qdd2, qdd3]
  row 3: [qdd2, qdd3]
  row 4: [qdd4]                      (user: only the stable qdd4 term)
  row 5: [qdd5]
  row 6: [qdd6]
Sensitivity: a symmetric 2x2 block for rows 2/3 (shared off-diagonal)
is fitted and reported alongside the unconstrained version.

r_fast (optional, fitted AFTER explicit terms on the remaining motion
residual): ridge regression per joint on shared features
  sin/cos(q1..6), qd, |qd|, qd_i*qd_j (i<j, except 1-2 which is in h),
  tanh(20qd), tanh(2qd)  (50 features + intercept).
Inclusion rule (pre-registered, test untouched): leave-one-recording-out
CV on train motion samples (recording = seg_id // 6, 11 recordings).
r_fast is delivered only if CV mean motion RMSE improves >= 2% over
explicit-only; otherwise it is reported as a diagnostic only.

Outputs: stage1_motion.json, stage1_motion_params.npz,
  stage1_e1.npz (e^(1) after delivered stage 1, train + test),
  stage1_ablation.svg, stage1_motion_rmse.svg
"""
import json
import os

import numpy as np

WORK = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(WORK, 'filtered_causal.npz'))
r0 = np.load(os.path.join(WORK, 'residuals.npz'))


def explicit_features(q, qd, qdd):
    """Return list of per-row feature matrices (N, k_row)."""
    prod12 = qd[0] * qd[1]
    return [
        np.column_stack([qdd[0], prod12, np.sin(q[1]) * prod12, np.cos(q[1]) * prod12]),
        np.column_stack([qdd[1], qdd[2]]),
        np.column_stack([qdd[1], qdd[2]]),
        qdd[3:4].T,
        qdd[4:5].T,
        qdd[5:6].T,
    ]


def fast_features(q, qd):
    feats = []
    names = []
    for j in range(6):
        feats.append(np.sin(q[j])); names.append(f'sin_q{j+1}')
        feats.append(np.cos(q[j])); names.append(f'cos_q{j+1}')
    for j in range(6):
        feats.append(qd[j]); names.append(f'qd{j+1}')
    for j in range(6):
        feats.append(np.abs(qd[j])); names.append(f'abs_qd{j+1}')
    for i in range(6):
        for k in range(i + 1, 6):
            if (i, k) == (0, 1):
                continue
            feats.append(qd[i] * qd[k]); names.append(f'qd{i+1}_x_qd{k+1}')
    for j in range(6):
        feats.append(np.tanh(20 * qd[j])); names.append(f'tanh20_qd{j+1}')
    for j in range(6):
        feats.append(np.tanh(2 * qd[j])); names.append(f'tanh2_qd{j+1}')
    return np.column_stack(feats), names


def fit_ls(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def apply_explicit(feats, coefs, N):
    out = np.zeros((6, N))
    for j in range(6):
        out[j] = feats[j] @ coefs[j]
    return out


def rmse_by_regime(resid, slow):
    def rm(a):
        return float(np.sqrt(np.mean(a ** 2))) if a.size else float('nan')
    return {
        'overall': [rm(resid[j]) for j in range(6)],
        'motion': [rm(resid[j, ~slow]) for j in range(6)],
        'slow': [rm(resid[j, slow]) for j in range(6)],
    }


data = {}
for split in ['train', 'test']:
    q, qd, qdd = z[f'{split}_q'], z[f'{split}_qd'], z[f'{split}_qdd']
    e0 = r0[f'{split}_resid']
    slow = np.max(np.abs(qd), axis=0) < 0.05
    data[split] = {'q': q, 'qd': qd, 'qdd': qdd, 'e0': e0, 'slow': slow,
                   'feats': explicit_features(q, qd, qdd),
                   'fast': fast_features(q, qd)}

# --- explicit fit on train motion samples ---
tr = data['train']
mov = ~tr['slow']
coefs = []
for j in range(6):
    coefs.append(fit_ls(tr['feats'][j][mov], tr['e0'][j, mov]))
print('explicit coefs:')
print(' row1 [dM11, a, b, c] =', np.round(coefs[0], 3))
print(' row2 [dM22, dM23]    =', np.round(coefs[1], 3))
print(' row3 [dM32, dM33]    =', np.round(coefs[2], 3))
print(' row4 [dM44]          =', np.round(coefs[3], 3))
print(' row5 [dM55]          =', np.round(coefs[4], 3))
print(' row6 [dM66]          =', np.round(coefs[5], 3))

# --- symmetric 2x2 sensitivity (rows 2/3, shared off-diagonal) ---
X2 = tr['feats'][1][mov]
X3 = tr['feats'][2][mov]
A = np.zeros((2 * mov.sum(), 3))
A[:mov.sum(), 0] = X2[:, 0]
A[:mov.sum(), 1] = X2[:, 1]
A[mov.sum():, 1] = X3[:, 0]
A[mov.sum():, 2] = X3[:, 1]
y_sym = np.concatenate([tr['e0'][1, mov], tr['e0'][2, mov]])
beta_sym = fit_ls(A, y_sym)
print(' symmetric 2x2: [dM22, shared, dM33] =', np.round(beta_sym, 3))


def eval_model(name, resid_fn):
    out = {}
    for split in ['train', 'test']:
        d = data[split]
        resid = resid_fn(d)
        out[split] = rmse_by_regime(resid, d['slow'])
        for regime in ['overall', 'motion', 'slow']:
            v = np.array(out[split][regime])
            print(f' {name} {split} {regime}: mean {v.mean():.3f}  per-joint {np.round(v, 3)}')
    return out


def resid_explicit(d):
    return d['e0'] - apply_explicit(d['feats'], coefs, d['e0'].shape[1])


def resid_sym(d):
    r = resid_explicit(d)
    # replace rows 2/3 explicit part with symmetric version
    r[1] = d['e0'][1] - (beta_sym[0] * d['feats'][1][:, 0] + beta_sym[1] * d['feats'][1][:, 1])
    r[2] = d['e0'][2] - (beta_sym[1] * d['feats'][2][:, 0] + beta_sym[2] * d['feats'][2][:, 1])
    return r


print('\n-- phy (PSD, e0) --')
met_phy = eval_model('phy', lambda d: d['e0'])
print('\n-- phy+motion explicit --')
met_exp = eval_model('phy+motion-explicit', resid_explicit)
print('\n-- phy+motion symmetric-2x2 --')
met_sym = eval_model('phy+motion-sym', resid_sym)

# --- r_fast: recording-block CV on train motion samples ---
rec = (z['train_seg_id'] // 6).astype(int)
Xf, fast_names = tr['fast']
e_exp_tr = resid_explicit(tr)
alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
cv = {}
for alpha in alphas:
    errs = []
    for fold in np.unique(rec):
        te_m = (rec == fold) & mov
        tr_m = (rec != fold) & mov
        mu = Xf[tr_m].mean(axis=0)
        sd = Xf[tr_m].std(axis=0)
        sd[sd == 0] = 1.0
        Xs = (Xf - mu) / sd
        pred = np.zeros((6, te_m.sum()))
        for j in range(6):
            y = e_exp_tr[j, tr_m] - e_exp_tr[j, tr_m].mean()
            beta = np.linalg.solve(Xs[tr_m].T @ Xs[tr_m] + alpha * np.eye(Xf.shape[1]),
                                   Xs[tr_m].T @ y)
            pred[j] = Xs[te_m] @ beta + e_exp_tr[j, tr_m].mean()
        errs.append(np.mean((e_exp_tr[:, te_m] - pred) ** 2))
    cv[alpha] = float(np.sqrt(np.mean(errs)))
base_cv = None
errs0 = []
for fold in np.unique(rec):
    te_m = (rec == fold) & mov
    errs0.append(np.mean(e_exp_tr[:, te_m] ** 2))
base_cv = float(np.sqrt(np.mean(errs0)))
print(f'\nCV motion RMSE explicit-only: {base_cv:.4f}')
for a, v in cv.items():
    print(f' CV motion RMSE explicit+fast alpha={a}: {v:.4f} (improvement {(base_cv - v) / base_cv * 100:+.2f}%)')
best_alpha = min(cv, key=cv.get)
fast_delivered = (base_cv - cv[best_alpha]) / base_cv >= 0.02
print(f'best alpha {best_alpha}; fast delivered = {fast_delivered} (rule: >=2% CV improvement)')

# fit fast on full train motion samples (for reporting / delivery if rule passes)
mu = Xf[mov].mean(axis=0)
sd = Xf[mov].std(axis=0)
sd[sd == 0] = 1.0
Xs = (Xf - mu) / sd
fast_beta = np.zeros((6, Xf.shape[1]))
fast_mean = np.zeros(6)
for j in range(6):
    fast_mean[j] = e_exp_tr[j, mov].mean()
    y = e_exp_tr[j, mov] - fast_mean[j]
    fast_beta[j] = np.linalg.solve(Xs[mov].T @ Xs[mov] + best_alpha * np.eye(Xf.shape[1]),
                                   Xs[mov].T @ y)


def resid_fast(d):
    Xd = (d['fast'][0] - mu) / sd
    return resid_explicit(d) - (fast_beta @ Xd.T + fast_mean[:, None])


print('\n-- phy+motion explicit+fast (full-train fit) --')
met_fast = eval_model('phy+motion-explicit+fast', resid_fast)

delivered = 'explicit+fast' if fast_delivered else 'explicit'
print(f'\nDELIVERED stage 1: {delivered}')


def resid_delivered(d):
    return resid_fast(d) if fast_delivered else resid_explicit(d)


met_del = eval_model('phy+motion-DELIVERED', resid_delivered)

out = {
    'motion_definition': 'motion = max_j |qd_j| >= 0.05 rad/s; slow = complement',
    'explicit_coefs': {
        'row1_dM11_a_b_c': coefs[0].tolist(),
        'row2_dM22_dM23': coefs[1].tolist(),
        'row3_dM32_dM33': coefs[2].tolist(),
        'row4_dM44': coefs[3].tolist(),
        'row5_dM55': coefs[4].tolist(),
        'row6_dM66': coefs[5].tolist(),
        'symmetric_2x2_dM22_shared_dM33': beta_sym.tolist(),
    },
    'fast_cv': {'explicit_only_cv_motion_rmse': base_cv,
                'alpha_cv_motion_rmse': {str(k): v for k, v in cv.items()},
                'best_alpha': best_alpha, 'delivered': fast_delivered},
    'metrics': {'phy': met_phy, 'explicit': met_exp, 'symmetric': met_sym,
                'explicit_fast': met_fast, 'delivered': met_del},
    'delivered_model': delivered,
    'note': ('Stage 1 fitted on motion samples only; slow samples never used. '
             'e^(1) = delivered stage-1 residual, saved for stage 2.'),
}
json.dump(out, open(os.path.join(WORK, 'stage1_motion.json'), 'w'), indent=1)
e1 = {}
for split in ['train', 'test']:
    e1[f'{split}_explicit'] = resid_explicit(data[split])
    e1[f'{split}_delivered'] = resid_delivered(data[split])
np.savez_compressed(os.path.join(WORK, 'stage1_e1.npz'), **e1)
np.savez_compressed(os.path.join(WORK, 'stage1_motion_params.npz'),
                    explicit_coefs=np.array([np.pad(c, (0, 4 - len(c))) for c in coefs]),
                    fast_beta=fast_beta, fast_mu=mu, fast_sd=sd, fast_mean=fast_mean,
                    fast_delivered=np.array([fast_delivered]),
                    symmetric_2x2=beta_sym)
print('saved stage1_motion.json, stage1_e1.npz, stage1_motion_params.npz')

# plots (SVG, per user instruction)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, split in zip(axes, ['train', 'test']):
    x = np.arange(6)
    for name, met, off, col in [('phy', met_phy, -0.25, 'C0'),
                                ('+motion explicit', met_exp, 0.0, 'C1'),
                                ('+motion +fast', met_fast, 0.25, 'C2')]:
        ax.bar(x + off, met[split]['motion'], 0.25, label=name, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f'J{j+1}' for j in range(6)])
    ax.set_ylabel('motion-sample RMSE (Nm)')
    ax.set_title(f'{split}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('Stage 1 ablation: motion-sample RMSE per joint (motion samples only in fitting)')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'stage1_motion_rmse.svg'))
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
for ax, regime in zip(axes, ['overall', 'motion', 'slow']):
    x = np.arange(2)
    for name, met, col in [('phy', met_phy, 'C0'), ('phy+motion', met_del, 'C1')]:
        vals = [np.mean(met[s][regime]) for s in ['train', 'test']]
        ax.bar(x, vals, 0.5, label=name, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(['train', 'test'])
    ax.set_ylabel('mean RMSE (Nm)')
    ax.set_title(f'{regime}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('Stage 1 ablation: mean RMSE by regime (delivered stage-1 model)')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'stage1_ablation.svg'))
plt.close(fig)
print('saved stage1_motion_rmse.svg, stage1_ablation.svg')

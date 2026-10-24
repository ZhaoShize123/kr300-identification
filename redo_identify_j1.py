"""Joint-1 inertia correction on top of the PSD model, re-identified.

Model: tau_j = PSD-model prediction (pi fixed at its PSD-projected values,
r / friction / hydraulic refit) + d * qdd_meas,1 on joint 1 only.
d is deliberately kept OUTSIDE the PSD projection: it is an effective
inertia term (e.g. reflected motor-rotor inertia, which is not part of
any link spatial inertia), so projecting it away would recreate the
distortion it is meant to fix. If d were put inside the pi fit before
projection, the projection would distort it again.

Procedure:
  1. post-hoc d: freeze final_psd theta/r, scalar LS of J1 residual on
     qdd_meas,1 (train only).
  2. joint refit: alternate (a) refit_fric_hyd_r with pi fixed and
     tau_1 adjusted by -d*qdd_1, (b) scalar LS for d given the rest,
     until convergence. r_1 stays 1.
  3. Evaluate frozen parameters on train and test (test untouched
     during fitting), compare with PSD baseline and pre-PSD baseline.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identify import compute_Y, predict, refit_fric_hyd_r, metrics  # noqa: E402

WORK = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(WORK, 'filtered_causal.npz'))
fp = np.load(os.path.join(WORK, 'final_psd_params.npz'))
theta0, r0, pi_psd = fp['theta'].copy(), fp['r'].copy(), fp['pi'].copy()

q_tr, qd_tr, qdd_tr, tau_tr = z['train_q'], z['train_qd'], z['train_qdd'], z['train_tau']
q_te, qd_te, qdd_te, tau_te = z['test_q'], z['test_qd'], z['test_qdd'], z['test_tau']

print('computing Y train/test ...', flush=True)
Y_tr = compute_Y(q_tr, qd_tr, qdd_tr)
Y_te = compute_Y(q_te, qd_te, qdd_te)


def pred_with_d(Y, r, theta, q, qd, qdd, d):
    p = predict(Y, r, theta, q, qd)
    p[0] = p[0] + d * qdd[0]
    return p


def fit_d(resid1, qdd1):
    den = float(qdd1 @ qdd1)
    return float(np.clip((qdd1 @ resid1) / den, -100.0, 100.0)) if den > 0 else 0.0


# 1. post-hoc d, everything else frozen at final_psd
pred_tr0 = predict(Y_tr, r0, theta0, q_tr, qd_tr)
d_posthoc = fit_d(tau_tr[0] - pred_tr0[0], qdd_tr[0])
print(f'post-hoc d = {d_posthoc:.4f}', flush=True)

# 2. joint refit
theta, r, d = theta0.copy(), r0.copy(), d_posthoc
for it in range(12):
    tau_adj = tau_tr.copy()
    tau_adj[0] = tau_adj[0] - d * qdd_tr[0]
    theta_new, r_new = refit_fric_hyd_r(Y_tr, pi_psd, r.copy(), q_tr, qd_tr, tau_adj)
    # refit keeps pi fixed (returns theta with pi_psd); take its fric/hyd
    theta = theta_new
    r = r_new
    pred_wo = predict(Y_tr, r, theta, q_tr, qd_tr)
    d_new = fit_d(tau_tr[0] - pred_wo[0], qdd_tr[0])
    print(f'iter {it}: d {d:.4f} -> {d_new:.4f}, r {np.round(r, 3)}', flush=True)
    if abs(d_new - d) < 1e-6:
        d = d_new
        break
    d = d_new

pred_tr = pred_with_d(Y_tr, r, theta, q_tr, qd_tr, qdd_tr, d)
pred_te = pred_with_d(Y_te, r, theta, q_te, qd_te, qdd_te, d)
pred_te_posthoc = pred_with_d(Y_te, r0, theta0, q_te, qd_te, qdd_te, d_posthoc)
pred_tr_posthoc = pred_with_d(Y_tr, r0, theta0, q_tr, qd_tr, qdd_tr, d_posthoc)

met_tr, met_te = metrics(tau_tr, pred_tr), metrics(tau_te, pred_te)
met_tr_ph, met_te_ph = metrics(tau_tr, pred_tr_posthoc), metrics(tau_te, pred_te_posthoc)
print('joint-refit train mean', met_tr['mean'])
print('joint-refit test  mean', met_te['mean'])
print('joint-refit test  J1  ', met_te['joint1'])
print('posthoc     test  mean', met_te_ph['mean'], 'J1', met_te_ph['joint1'])

# residual correlations for J1 after correction
for name, res, qdd in [('train', tau_tr[0] - pred_tr[0], qdd_tr[0]),
                       ('test', tau_te[0] - pred_te[0], qdd_te[0])]:
    print(f'J1 corrected {name}: corr res~qdd {np.corrcoef(res, qdd)[0, 1]:+.3f}, '
          f'slope {np.polyfit(qdd, res, 1)[0]:+.3f}')

out = {
    'd_j1_inertia_correction': d,
    'd_posthoc_only': d_posthoc,
    'r': r.tolist(), 'theta': theta.tolist(), 'pi_psd': pi_psd.tolist(),
    'metrics_train': met_tr, 'metrics_test': met_te,
    'metrics_train_posthoc': met_tr_ph, 'metrics_test_posthoc': met_te_ph,
    'note': ('tau_1 += d * qdd_meas,1 added on top of the PSD model; pi kept at '
             'its PSD-projected values, r/friction/hydraulic refit jointly '
             'with d on train only. d is an effective (e.g. rotor) inertia '
             'term, not part of link spatial inertia, hence not PSD-projected.'),
}
json.dump(out, open(os.path.join(WORK, 'j1_correction.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(WORK, 'j1_correction_params.npz'),
                    theta=theta, r=r, d=np.array([d]),
                    test_pred=pred_te, train_pred=pred_tr)
print('saved j1_correction.json / j1_correction_params.npz')

# plots: test fit (6 panels, corrected) + J1 residual vs qdd before/after
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
t = np.arange(q_te.shape[1]) * 0.1
for j in range(6):
    ax = axes[j]
    ax.plot(t, tau_te[j], label='measured', linewidth=0.7)
    ax.plot(t, pred_te[j], label='predicted (PSD + J1 correction)', linewidth=0.7)
    ax.set_ylabel(f'joint {j+1} torque (Nm)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel('time (s, test recording, 6 trajectories concatenated)')
fig.suptitle('KR300 inverse model: test-set torque fit (PSD + J1 inertia correction)')
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'test_fit_j1_corrected.png'), dpi=150)
plt.close(fig)

res_tr_psd = np.load(os.path.join(WORK, 'residuals.npz'))['train_resid']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, res, title in [(axes[0], res_tr_psd[0], 'J1 residual vs qdd1: PSD model (before)'),
                       (axes[1], tau_tr[0] - pred_tr[0], 'J1 residual vs qdd1: + correction (after)')]:
    ax.scatter(qdd_tr[0][::7], res[::7], s=4, alpha=0.4)
    xs = np.linspace(qdd_tr[0].min(), qdd_tr[0].max(), 50)
    ax.plot(xs, np.polyfit(qdd_tr[0], res, 1)[0] * xs + np.polyfit(qdd_tr[0], res, 1)[1],
            color='r', linewidth=1.5)
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel('qdd1 meas (rad/s^2)')
    ax.set_ylabel('residual (Nm)')
    ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(WORK, 'j1_residual_vs_qdd.png'), dpi=150)
plt.close(fig)
print('saved test_fit_j1_corrected.png, j1_residual_vs_qdd.png')

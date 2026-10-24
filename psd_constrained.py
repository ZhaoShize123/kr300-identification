"""PSD-constrained refit (replaces the post-hoc projection as final model).

Why: post-hoc PSD projection minimises Frobenius distance in spatial-
inertia space, not torque error, and it degraded joint-1 test R^2 from
0.898 to 0.425. The spec asks for physically feasible inertias, so the
right object is the torque-optimal parameter vector *inside* the PSD
cone, found by projected gradient (FISTA) on the torque loss with the
per-link PSD projection as the projector, with r held at the pre-PSD
values (r_4/r_6 are only weakly identified and drift to bounds if freed)
and friction/hydraulic refit afterwards with pi fixed.

Outputs: work/psd_constrained.json, work/psd_constrained_params.npz,
         work/test_fit_psd_constrained.png
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from identify import (compute_Y, friction_basis, hyd_basis, metrics,
                      predict, psd_project, spatial_from_pi,
                      pi_from_spatial)  # noqa: E402

WORK = os.path.dirname(os.path.abspath(__file__))


def psd_proj_pi(pi):
    out = pi.copy()
    for i in range(6):
        S = spatial_from_pi(pi[i * 10:(i + 1) * 10])
        S = 0.5 * (S + S.T)
        w, V = np.linalg.eigh(S)
        w_clip = np.maximum(w, 1e-9 * max(w[-1], 1e-12))
        S_new = (V * w_clip) @ V.T
        S_new = 0.5 * (S_new + S_new.T)
        out[i * 10:(i + 1) * 10] = pi_from_spatial(S_new)
    return out


def box_clip_pi(pi):
    out = pi.copy()
    for i in range(6):
        b = i * 10
        out[b] = np.clip(out[b], 0.0, 600.0)
        out[b + 1:b + 4] = np.clip(out[b + 1:b + 4], -600.0, 600.0)
        out[b + 4] = np.clip(out[b + 4], 0.0, 1000.0)
        out[b + 5] = np.clip(out[b + 5], -500.0, 500.0)
        out[b + 6] = np.clip(out[b + 6], 0.0, 1000.0)
        out[b + 7] = np.clip(out[b + 7], -500.0, 500.0)
        out[b + 8] = np.clip(out[b + 8], -500.0, 500.0)
        out[b + 9] = np.clip(out[b + 9], 0.0, 1000.0)
    return out


def min_eig(pi):
    vals = []
    for i in range(6):
        S = spatial_from_pi(pi[i * 10:(i + 1) * 10])
        vals.append(float(np.linalg.eigvalsh(0.5 * (S + S.T))[0]))
    return vals


def main():
    z = np.load(os.path.join(WORK, "filtered_causal.npz"))
    q_tr, qd_tr, qdd_tr, tau_tr = z["train_q"], z["train_qd"], z["train_qdd"], z["train_tau"]
    q_te, qd_te, qdd_te, tau_te = z["test_q"], z["test_qd"], z["test_qdd"], z["test_tau"]
    p = np.load(os.path.join(WORK, "identification_params.npz"))
    theta_pre = p["theta_pre_psd"]
    r = p["r_pre_psd"].copy()
    print("computing Y ...", flush=True)
    Y_tr = compute_Y(q_tr, qd_tr, qdd_tr)
    Y_te = compute_Y(q_te, qd_te, qdd_te)
    # Residual after fixed friction/hyd, design for pi given r.
    resid = tau_tr.copy()
    for j in range(6):
        resid[j] -= friction_basis(qd_tr[j]) @ theta_pre[60 + 4 * j:60 + 4 * j + 4]
    resid[1] -= hyd_basis(q_tr[1]) @ theta_pre[84:]
    A = np.zeros((60, 60))
    b = np.zeros(60)
    for j in range(6):
        D = r[j] * Y_tr[:, j, :]
        A += D.T @ D
        b += D.T @ resid[j]
    L = np.linalg.eigvalsh(A)[-1]
    pi = psd_proj_pi(theta_pre[:60])
    pi = box_clip_pi(pi)
    y = pi.copy()
    t = 1.0
    for it in range(800):
        grad = A @ y - b
        xn = box_clip_pi(psd_proj_pi(y - grad / L))
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = xn + ((t - 1.0) / tn) * (xn - pi)
        pi, t = xn, tn
        if it % 100 == 0 or it == 799:
            theta_tmp = np.concatenate([pi, theta_pre[60:]])
            pred = predict(Y_tr, r, theta_tmp, q_tr, qd_tr)
            rmse = float(np.sqrt(np.mean((tau_tr - pred) ** 2)))
            print(f"iter {it}: train RMSE (pi only, fric/hyd fixed) {rmse:.4f}, "
                  f"min eig {np.round(min_eig(pi), 4)}", flush=True)
    # Refit friction/hyd with pi fixed, r fixed.
    # Build normal equations for fric/hyd only.
    dim = 28
    A_fh = np.zeros((dim, dim))
    b_fh = np.zeros(dim)
    for j in range(6):
        F = friction_basis(qd_tr[j])
        res = tau_tr[j] - r[j] * (Y_tr[:, j, :] @ pi)
        A_fh[4 * j:4 * j + 4, 4 * j:4 * j + 4] += F.T @ F
        b_fh[4 * j:4 * j + 4] += F.T @ res
        if j == 1:
            H = hyd_basis(q_tr[1])
            A_fh[4 * j:4 * j + 4, 24:] += F.T @ H
            A_fh[24:, 4 * j:4 * j + 4] += H.T @ F
            A_fh[24:, 24:] += H.T @ H
            b_fh[24:] += H.T @ res
    # Bounded solve via simple projected gradient (same FISTA pattern, no PSD)
    lo = np.empty(dim)
    hi = np.empty(dim)
    for j in range(6):
        lo[4 * j:4 * j + 4] = [-10.0, 0.0, 0.0, -20.0]
        hi[4 * j:4 * j + 4] = [10.0, 20.0, 20.0, 20.0]
    lo[24:] = -40.0
    hi[24:] = 40.0
    Lf = np.linalg.eigvalsh(A_fh)[-1]
    x = np.clip(np.linalg.solve(A_fh + 1e-9 * np.eye(dim), b_fh), lo, hi)
    yy = x.copy()
    tt = 1.0
    for _ in range(1000):
        xn = np.clip(yy - (A_fh @ yy - b_fh) / Lf, lo, hi)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tt * tt))
        yy = xn + ((tt - 1.0) / tn) * (xn - x)
        x, tt = xn, tn
    theta = np.concatenate([pi, x[:24], x[24:]])
    pred_tr = predict(Y_tr, r, theta, q_tr, qd_tr)
    pred_te = predict(Y_te, r, theta, q_te, qd_te)
    met_tr = metrics(tau_tr, pred_tr)
    met_te = metrics(tau_te, pred_te)
    print("PSD-constrained train", met_tr["mean"], "test", met_te["mean"], flush=True)
    print("min eig", min_eig(pi), flush=True)
    out = {
        "pi": pi.tolist(), "theta": theta.tolist(), "r": r.tolist(),
        "friction": theta[60:84].tolist(), "hydraulic": theta[84:].tolist(),
        "min_eig_per_link": min_eig(pi),
        "metrics_train": met_tr, "metrics_test": met_te,
    }
    with open(os.path.join(WORK, "psd_constrained.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(
        os.path.join(WORK, "psd_constrained_params.npz"),
        pi=pi, theta=theta, r=r, test_pred=pred_te)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    tt_axis = np.arange(q_te.shape[1]) * 0.1
    for j in range(6):
        ax = axes[j]
        ax.plot(tt_axis, tau_te[j], label="measured", linewidth=0.7)
        ax.plot(tt_axis, pred_te[j], label="predicted", linewidth=0.7)
        ax.set_ylabel(f"joint {j+1} torque (Nm)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s, test recording, 6 trajectories concatenated)")
    fig.suptitle("KR300 inverse model: test-set torque fit (PSD-constrained, frozen parameters)")
    fig.tight_layout()
    fig.savefig(os.path.join(WORK, "test_fit_psd_constrained.png"), dpi=150)
    print("wrote psd_constrained.json and test_fit_psd_constrained.png", flush=True)


if __name__ == "__main__":
    main()

"""Newton-Euler identification for the KR300 benchmark.

Model (motor-torque units, measurement convention):
    tau_j = r_j * (Y_j(q,qd,qdd) @ pi) + friction_j(qd_j) + hyd_j(q2)
  * Y: pinocchio joint-torque regressor (6x60) for the KR300 MDH model
    (kr300_model.py, HAL nominal DH, validated against KUKA datasheet
    dimensions A=2826mm / C=2496mm).
  * pi: 60 barycentric parameters, 10 per link, in pinocchio body frames.
    Torque prediction is invariant to a common positive scale of pi
    together with r, so r_1 is fixed to 1: pi is therefore the physical
    parameter vector divided by the (unknown) joint-1 gear ratio u_1.
    Relative gear ratios r_j = u_1 / u_j are estimated.
  * friction_j = f0 + fv*qd + fc*tanh(20*qd) + fa*tanh(2*qd), qd meas. rad/s.
  * hyd (joint 2 only) = Fourier basis in meas. q2: sin/cos(q2), sin/cos(2q2).
  * Phase lags are 0 for all joints: the phase-calibration CV (see
    phase_calibration.json) is flat to <0.001 Nm over 0..20 ms, i.e. the
    lag is unidentifiable at this 2-Hz bandwidth, so no shift is applied
    rather than inventing a per-joint lag from noise.
  * Fitting: alternating bounded least squares (FISTA on normal equations)
    for (pi, friction, hyd) given r, and scalar least squares for r given
    the rest. Bounds: m in [0,600] kg, |mc| <= 600 kg m, diagonal I in
    [0,1000] kg m^2, |off-diag I| <= 500, friction/hyd as in code.
  * PSD projection: per link, build the 6x6 spatial inertia S from pi,
    symmetrise, clip eigenvalues at >= 1e-9 * max eigenvalue, rebuild,
    extract pi back (m = mean diag of the linear block, mc from the
    skew block, I from the angular block). After projection, r/friction/
    hyd are refit with pi held fixed, then test metrics are recomputed.
  * Metrics follow the benchmark PDF: per-joint NRMSE = RMSE / std(tau)
    and R^2 = 1 - SSE / sum((tau - mean(tau))^2), on the frozen test set.

Outputs: work/identification.json, work/identification_params.npz,
         work/test_fit.png
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kr300_model import build_model, meas_to_dh  # noqa: E402

import pinocchio as pin  # noqa: E402

WORK = os.path.dirname(os.path.abspath(__file__))
SIGN = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0])


def compute_Y(q, qd, qdd):
    model = build_model()
    data = model.createData()
    # q/qd/qdd are (6, N); meas_to_dh expects (..., 6), so transpose.
    q_dh = meas_to_dh(q.T).T
    qd_dh = (qd.T * SIGN).T
    qdd_dh = (qdd.T * SIGN).T
    N = q.shape[1]
    Y = np.empty((N, 6, 60))
    for n in range(N):
        Y[n] = pin.computeJointTorqueRegressor(
            model, data, q_dh[:, n], qd_dh[:, n], qdd_dh[:, n]
        )
    return Y


def friction_basis(qd_j):
    return np.stack(
        [np.ones_like(qd_j), qd_j, np.tanh(20.0 * qd_j), np.tanh(2.0 * qd_j)],
        axis=1,
    )


def hyd_basis(q2):
    return np.stack(
        [np.sin(q2), np.cos(q2), np.sin(2 * q2), np.cos(2 * q2)], axis=1
    )


def build_normal_eq(Y, r, q, qd, tau):
    """Normal equations for theta = [pi(60), fric(24), hyd(4)] given r.

    Returns (A, b, theta_lo, theta_hi). Design is stacked over joints.
    """
    N = q.shape[1]
    # Per-joint blocks
    pi_blocks = []
    for j in range(6):
        pi_blocks.append(r[j] * Y[:, j, :])  # (N,60)
    fric_blocks = [friction_basis(qd[j]) for j in range(6)]
    hyd = hyd_basis(q[1])
    # Assemble A = X^T X, b = X^T y via block accumulation.
    dim = 60 + 24 + 4
    A = np.zeros((dim, dim))
    b = np.zeros(dim)
    # pi-pi
    for j in range(6):
        A[:60, :60] += pi_blocks[j].T @ pi_blocks[j]
        b[:60] += pi_blocks[j].T @ tau[j]
    # pi-fric, fric-fric, pi-hyd, fric-hyd, hyd-hyd
    for j in range(6):
        fj = 60 + 4 * j
        A[:60, fj:fj + 4] += pi_blocks[j].T @ fric_blocks[j]
        A[fj:fj + 4, :60] += fric_blocks[j].T @ pi_blocks[j]
        A[fj:fj + 4, fj:fj + 4] += fric_blocks[j].T @ fric_blocks[j]
        b[fj:fj + 4] += fric_blocks[j].T @ tau[j]
        if j == 1:
            A[:60, 84:] += pi_blocks[j].T @ hyd
            A[84:, :60] += hyd.T @ pi_blocks[j]
            A[fj:fj + 4, 84:] += fric_blocks[j].T @ hyd
            A[84:, fj:fj + 4] += hyd.T @ fric_blocks[j]
            A[84:, 84:] += hyd.T @ hyd
            b[84:] += hyd.T @ tau[j]
    lo = np.empty(dim)
    hi = np.empty(dim)
    for i in range(6):
        base = i * 10
        lo[base] = 0.0
        hi[base] = 600.0  # mass
        lo[base + 1:base + 4] = -600.0
        hi[base + 1:base + 4] = 600.0  # mc
        # Ixx, Ixy, Iyy, Ixz, Iyz, Izz
        lo[base + 4] = 0.0
        hi[base + 4] = 1000.0
        lo[base + 5] = -500.0
        hi[base + 5] = 500.0
        lo[base + 6] = 0.0
        hi[base + 6] = 1000.0
        lo[base + 7] = -500.0
        hi[base + 7] = 500.0
        lo[base + 8] = -500.0
        hi[base + 8] = 500.0
        lo[base + 9] = 0.0
        hi[base + 9] = 1000.0
    for j in range(6):
        fj = 60 + 4 * j
        lo[fj] = -10.0
        hi[fj] = 10.0  # f0
        lo[fj + 1] = 0.0
        hi[fj + 1] = 20.0  # fv (viscous >= 0)
        lo[fj + 2] = 0.0
        hi[fj + 2] = 20.0  # fc (Coulomb >= 0)
        lo[fj + 3] = -20.0
        hi[fj + 3] = 20.0  # fa
    lo[84:] = -40.0
    hi[84:] = 40.0
    return A, b, lo, hi


def fista(A, b, lo, hi, ridge=1e-6, iters=2000):
    """Minimise 0.5 theta^T (A+ridge*diag(A)) theta - b^T theta over box."""
    d = np.diag(A).copy()
    d[d <= 0] = 1.0
    Ar = A + ridge * np.diag(d)
    L = np.linalg.eigvalsh(Ar)[-1]
    x = np.clip(np.linalg.solve(Ar + 1e-9 * np.eye(len(b)), b), lo, hi)
    y = x.copy()
    t = 1.0
    for _ in range(iters):
        grad = Ar @ y - b
        xn = np.clip(y - grad / L, lo, hi)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = xn + ((t - 1.0) / tn) * (xn - x)
        x, t = xn, tn
    return x


def predict(Y, r, theta, q, qd):
    pi = theta[:60]
    pred = np.empty((6, q.shape[1]))
    for j in range(6):
        fj = 60 + 4 * j
        p = r[j] * (Y[:, j, :] @ pi) + friction_basis(qd[j]) @ theta[fj:fj + 4]
        if j == 1:
            p = p + hyd_basis(q[1]) @ theta[84:]
        pred[j] = p
    return pred


def metrics(tau, pred):
    out = {}
    for j in range(6):
        err = tau[j] - pred[j]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        nrmse = float(rmse / np.std(tau[j]))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((tau[j] - np.mean(tau[j])) ** 2))
        out[f"joint{j+1}"] = {"rmse_Nm": rmse, "nrmse": nrmse,
                             "r2": float(1.0 - ss_res / ss_tot)}
    out["mean"] = {"rmse_Nm": float(np.mean([out[f"joint{j+1}"]["rmse_Nm"] for j in range(6)])),
                   "nrmse": float(np.mean([out[f"joint{j+1}"]["nrmse"] for j in range(6)])),
                   "r2": float(np.mean([out[f"joint{j+1}"]["r2"] for j in range(6)]))}
    return out


def spatial_from_pi(pi_link):
    m, mcx, mcy, mcz, ixx, ixy, iyy, ixz, iyz, izz = pi_link
    mc = np.array([mcx, mcy, mcz])
    # skew(mc)
    K = np.array([[0, -mcz, mcy], [mcz, 0, -mcx], [-mcy, mcx, 0]])
    S = np.zeros((6, 6))
    S[:3, :3] = m * np.eye(3)
    S[:3, 3:] = -K
    S[3:, :3] = K
    S[3:, 3:] = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    return S


def pi_from_spatial(S):
    m = float(np.mean([S[0, 0], S[1, 1], S[2, 2]]))
    mc = np.array([S[5, 1], S[3, 2], S[4, 0]])
    I = np.array([[S[3, 3], S[3, 4], S[3, 5]],
                  [S[4, 3], S[4, 4], S[4, 5]],
                  [S[5, 3], S[5, 4], S[5, 5]]])
    return np.array([m, mc[0], mc[1], mc[2], I[0, 0], I[0, 1], I[1, 1],
                     I[0, 2], I[1, 2], I[2, 2]])


def psd_project(pi):
    pi_new = pi.copy()
    info = []
    for i in range(6):
        S = spatial_from_pi(pi[i * 10:(i + 1) * 10])
        S = 0.5 * (S + S.T)
        w, V = np.linalg.eigh(S)
        w_clip = np.maximum(w, 1e-9 * max(w[-1], 1e-12))
        S_new = (V * w_clip) @ V.T
        S_new = 0.5 * (S_new + S_new.T)
        pi_new[i * 10:(i + 1) * 10] = pi_from_spatial(S_new)
        info.append({"link": i + 1, "eig_min_before": float(w[0]),
                     "eig_max": float(w[-1]),
                     "n_clipped": int(np.sum(w < 1e-9 * max(w[-1], 1e-12))),
                     "fro_change": float(np.linalg.norm(S_new - S))})
    return pi_new, info


def refit_fric_hyd_r(Y, pi, r, q, qd, tau, iters=8):
    """Hold pi fixed; alternate scalar r and bounded friction/hyd fit."""
    # Friction/hyd fit with pi fixed: per-joint linear LS on residual,
    # solved by the same FISTA machinery restricted to those columns.
    theta_fh = np.zeros(28)
    for _ in range(iters):
        # r update
        for j in range(6):
            if j == 0:
                continue
            ypi = Y[:, j, :] @ pi
            resid = tau[j] - friction_basis(qd[j]) @ theta_fh[4 * j:4 * j + 4]
            if j == 1:
                resid = resid - hyd_basis(q[1]) @ theta_fh[24:]
            den = float(ypi @ ypi)
            r[j] = float(np.clip((ypi @ resid) / den, 0.2, 5.0)) if den > 0 else r[j]
        # friction/hyd update via normal equations (28 vars)
        dim = 28
        A = np.zeros((dim, dim))
        b = np.zeros(dim)
        for j in range(6):
            fj = 4 * j
            F = friction_basis(qd[j])
            resid = tau[j] - r[j] * (Y[:, j, :] @ pi)
            A[fj:fj + 4, fj:fj + 4] += F.T @ F
            b[fj:fj + 4] += F.T @ resid
            if j == 1:
                H = hyd_basis(q[1])
                A[fj:fj + 4, 24:] += F.T @ H
                A[24:, fj:fj + 4] += H.T @ F
                A[24:, 24:] += H.T @ H
                b[24:] += H.T @ resid
        lo = np.empty(dim)
        hi = np.empty(dim)
        for j in range(6):
            fj = 4 * j
            lo[fj:fj + 4] = [-10.0, 0.0, 0.0, -20.0]
            hi[fj:fj + 4] = [10.0, 20.0, 20.0, 20.0]
        lo[24:] = -40.0
        hi[24:] = 40.0
        theta_fh = fista(A, b, lo, hi, iters=1000)
    theta = np.concatenate([pi, theta_fh[:24], theta_fh[24:]])
    return theta, r


def main():
    z = np.load(os.path.join(WORK, "filtered_causal.npz"))
    q_tr, qd_tr, qdd_tr, tau_tr = z["train_q"], z["train_qd"], z["train_qdd"], z["train_tau"]
    q_te, qd_te, qdd_te, tau_te = z["test_q"], z["test_qd"], z["test_qdd"], z["test_tau"]
    print("computing Y train/test ...", flush=True)
    Y_tr = compute_Y(q_tr, qd_tr, qdd_tr)
    Y_te = compute_Y(q_te, qd_te, qdd_te)
    print("Y shapes", Y_tr.shape, Y_te.shape, flush=True)
    r = np.ones(6)
    theta = None
    for it in range(10):
        A, b, lo, hi = build_normal_eq(Y_tr, r, q_tr, qd_tr, tau_tr)
        theta_new = fista(A, b, lo, hi, iters=1500)
        # r update given theta
        pi = theta_new[:60]
        for j in range(6):
            if j == 0:
                continue
            ypi = Y_tr[:, j, :] @ pi
            resid = tau_tr[j] - friction_basis(qd_tr[j]) @ theta_new[60 + 4 * j:60 + 4 * j + 4]
            if j == 1:
                resid = resid - hyd_basis(q_tr[1]) @ theta_new[84:]
            den = float(ypi @ ypi)
            r[j] = float(np.clip((ypi @ resid) / den, 0.2, 5.0)) if den > 0 else r[j]
        change = np.max(np.abs(theta_new - theta)) if theta is not None else np.inf
        theta = theta_new
        pred_tr = predict(Y_tr, r, theta, q_tr, qd_tr)
        rmse = float(np.sqrt(np.mean((tau_tr - pred_tr) ** 2)))
        print(f"iter {it}: train RMSE {rmse:.4f} Nm, r={np.round(r, 3)}, "
              f"theta change {change:.2e}", flush=True)
        if change < 1e-4:
            break
    pred_tr = predict(Y_tr, r, theta, q_tr, qd_tr)
    pred_te = predict(Y_te, r, theta, q_te, qd_te)
    met_tr_pre = metrics(tau_tr, pred_tr)
    met_te_pre = metrics(tau_te, pred_te)
    print("pre-PSD train", met_tr_pre["mean"], "test", met_te_pre["mean"], flush=True)
    # PSD projection
    pi_psd, psd_info = psd_project(theta[:60])
    print("PSD info", psd_info, flush=True)
    theta_psd, r_psd = refit_fric_hyd_r(Y_tr, pi_psd, r.copy(), q_tr, qd_tr, tau_tr)
    pred_tr_psd = predict(Y_tr, r_psd, theta_psd, q_tr, qd_tr)
    pred_te_psd = predict(Y_te, r_psd, theta_psd, q_te, qd_te)
    met_tr_psd = metrics(tau_tr, pred_tr_psd)
    met_te_psd = metrics(tau_te, pred_te_psd)
    print("post-PSD train", met_tr_psd["mean"], "test", met_te_psd["mean"], flush=True)
    out = {
        "r_relative_gear": r_psd.tolist(),
        "r_pre_psd": r.tolist(),
        "pi_pre_psd": theta[:60].tolist(),
        "pi_post_psd": theta_psd[:60].tolist(),
        "friction_post_psd": theta_psd[60:84].tolist(),
        "hydraulic_post_psd": theta_psd[84:].tolist(),
        "friction_pre_psd": theta[60:84].tolist(),
        "hydraulic_pre_psd": theta[84:].tolist(),
        "psd_info": psd_info,
        "metrics_train_pre_psd": met_tr_pre,
        "metrics_test_pre_psd": met_te_pre,
        "metrics_train_post_psd": met_tr_psd,
        "metrics_test_post_psd": met_te_psd,
        "note": "pi is physical barycentric parameters divided by u_1 "
                "(joint-1 gear ratio, unknown); r_j = u_1/u_j. "
                "Official linear baseline (PDF Table 5, inverse, test): "
                "mean NRMSE 0.6085, mean R2 0.6200.",
    }
    with open(os.path.join(WORK, "identification.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    np.savez_compressed(
        os.path.join(WORK, "identification_params.npz"),
        pi_post_psd=theta_psd[:60], theta_post_psd=theta_psd, r_post_psd=r_psd,
        pi_pre_psd=theta[:60], theta_pre_psd=theta, r_pre_psd=r,
        test_pred_post_psd=pred_te_psd, test_pred_pre_psd=pred_te,
    )
    # Plot: test measured vs predicted (post-PSD), 6 joints, time in seconds
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    t = np.arange(q_te.shape[1]) * 0.1
    for j in range(6):
        ax = axes[j]
        ax.plot(t, tau_te[j], label="measured", linewidth=0.7)
        ax.plot(t, pred_te_psd[j], label="predicted", linewidth=0.7)
        ax.set_ylabel(f"joint {j+1} torque (Nm)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s, test recording, 6 trajectories concatenated)")
    fig.suptitle("KR300 inverse model: test-set torque fit (post-PSD, frozen parameters)")
    fig.tight_layout()
    fig.savefig(os.path.join(WORK, "test_fit.png"), dpi=150)
    print("wrote identification.json and test_fit.png", flush=True)


if __name__ == "__main__":
    main()

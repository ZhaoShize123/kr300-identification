"""Frozen three-stage cascade ablation (see freeze.json). Pure inverse dynamics."""
import json, os, sys, time, hashlib
import numpy as np, torch
REPO = os.environ.get('KR300_REPO', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); sys.path.insert(0, REPO)  # 仓库根目录（grok结果/three_stage 的上两级）
from identify import compute_Y, build_normal_eq, fista, predict, friction_basis, hyd_basis
from psd_constrained import psd_proj_pi, box_clip_pi
OUT = os.path.dirname(os.path.abspath(__file__)); torch.set_num_threads(8)
F = json.load(open(f'{OUT}/freeze.json'))
z = np.load(f'{REPO}/filtered_causal.npz')

def ys(split):
    p = f'{OUT}/Y_{split}.npz'
    if os.path.exists(p): d = np.load(p); return d['Yf'], d['Yc'], d['Yg']
    q, qd, qdd = z[split+'_q'], z[split+'_qd'], z[split+'_qdd']; Zr = np.zeros_like(q)
    Yf, Yc, Yg = compute_Y(q, qd, qdd), compute_Y(q, qd, Zr), compute_Y(q, Zr, Zr)
    np.savez(p, Yf=Yf, Yc=Yc, Yg=Yg); return Yf, Yc, Yg

D = {}
for s in ['train', 'test']:
    Yf, Yc, Yg = ys(s)
    D[s] = dict(q=z[s+'_q'], qd=z[s+'_qd'], qdd=z[s+'_qdd'], tau=z[s+'_tau'], seg=z[s+'_seg_id'], Yf=Yf, Yc=Yc, Yg=Yg)

def fit_stage1(idx):
    Y, q, qd, qdd, tau = (D['train'][k][..., idx] if k != 'Yf' else D['train'][k][idx] for k in ['Yf', 'q', 'qd', 'qdd', 'tau'])
    r = np.ones(6); theta = None
    for it in range(10):  # identify.py
        A, b, lo, hi = build_normal_eq(Y, r, q, qd, tau); tn_ = fista(A, b, lo, hi, iters=1500); pi = tn_[:60]
        for j in range(1, 6):
            ypi = Y[:, j, :] @ pi; res = tau[j] - friction_basis(qd[j]) @ tn_[60+4*j:64+4*j]
            if j == 1: res = res - hyd_basis(q[1]) @ tn_[84:]
            den = float(ypi @ ypi); r[j] = float(np.clip((ypi @ res)/den, 0.2, 5.0)) if den > 0 else r[j]
        ch = np.max(np.abs(tn_ - theta)) if theta is not None else np.inf; theta = tn_
        if ch < 1e-4: break
    th = theta; resid = tau.copy()  # psd_constrained.py
    for j in range(6): resid[j] -= friction_basis(qd[j]) @ th[60+4*j:64+4*j]
    resid[1] -= hyd_basis(q[1]) @ th[84:]
    A = np.zeros((60, 60)); b = np.zeros(60)
    for j in range(6): Dm = r[j]*Y[:, j, :]; A += Dm.T@Dm; b += Dm.T@resid[j]
    L = np.linalg.eigvalsh(A)[-1]; pi = box_clip_pi(psd_proj_pi(th[:60])); y = pi.copy(); t = 1.0
    for it in range(800):
        xn = box_clip_pi(psd_proj_pi(y - (A@y-b)/L)); tn = .5*(1+np.sqrt(1+4*t*t)); y = xn+((t-1)/tn)*(xn-pi); pi, t = xn, tn
    dim = 28; Af = np.zeros((dim, dim)); bf = np.zeros(dim)
    for j in range(6):
        Fb = friction_basis(qd[j]); res = tau[j]-r[j]*(Y[:, j, :]@pi)
        Af[4*j:4*j+4, 4*j:4*j+4] += Fb.T@Fb; bf[4*j:4*j+4] += Fb.T@res
        if j == 1:
            H = hyd_basis(q[1]); Af[4:8, 24:] += Fb.T@H; Af[24:, 4:8] += H.T@Fb; Af[24:, 24:] += H.T@H; bf[24:] += H.T@res
    lo = np.tile([-10., 0, 0, -20], 6).tolist()+[-40.]*4; hi = np.tile([10., 20, 20, 20], 6).tolist()+[40.]*4; lo, hi = np.array(lo), np.array(hi)
    Lf = np.linalg.eigvalsh(Af)[-1]; x = np.clip(np.linalg.solve(Af+1e-9*np.eye(dim), bf), lo, hi); yy = x.copy(); tt = 1.0
    for _ in range(1000):
        xn = np.clip(yy-(Af@yy-bf)/Lf, lo, hi); tn = .5*(1+np.sqrt(1+4*tt*tt)); yy = xn+((tt-1)/tn)*(xn-x); x, tt = xn, tn
    theta = np.concatenate([pi, x]); p = predict(Y, r, theta, q, qd)
    d = float(np.clip(qdd[0] @ (tau[0]-p[0]) / (qdd[0]@qdd[0]), -100, 100))
    return r, theta, d

def stage1_feats(split, idx, r, theta, d):
    S = D[split]; q, qd, qdd = S['q'][:, idx], S['qd'][:, idx], S['qdd'][:, idx]
    Yf, Yc, Yg = S['Yf'][idx], S['Yc'][idx], S['Yg'][idx]; pi = theta[:60]
    phi = np.zeros((6, 5, len(idx)))
    for j in range(6):
        phi[j, 0] = r[j]*((Yf[:, j]-Yc[:, j])@pi) + (d*qdd[0] if j == 0 else 0)
        phi[j, 1] = r[j]*((Yc[:, j]-Yg[:, j])@pi); phi[j, 2] = r[j]*(Yg[:, j]@pi)
        phi[j, 3] = friction_basis(qd[j]) @ theta[60+4*j:64+4*j]
        if j == 1: phi[j, 4] = hyd_basis(q[1]) @ theta[84:]
    tau1 = phi.sum(1)
    zf = np.concatenate([q, qd, qdd, tau1, phi.reshape(30, -1)], 0)  # (54,N)
    return tau1, zf

def masks(qd, seg):
    mv = np.max(np.abs(qd), 0) < 0.05; slow = np.zeros_like(mv); dwell = np.zeros_like(mv)
    n = len(mv); i = 0
    while i < n:
        if mv[i]:
            k = i
            while k < n and mv[k] and seg[k] == seg[i]: k += 1
            if k-i >= 10: slow[i:k] = True
            if k-i >= 30: dwell[i:k] = True
            i = k
        else: i += 1
    return slow, dwell

def to_seq(X, seg):  # X (C,N) -> list of (T,C)
    return [X[:, seg == s].T for s in np.unique(seg)]

def pad(seqs):
    T = max(len(s) for s in seqs); C = seqs[0].shape[1]; out = np.zeros((len(seqs), T, C), np.float32); m = np.zeros((len(seqs), T), bool)
    for i, s in enumerate(seqs): out[i, :len(s)] = s; m[i, :len(s)] = True
    return torch.tensor(out), torch.tensor(m)

class GRUM(torch.nn.Module):
    def __init__(s, cin): super().__init__(); s.g = torch.nn.GRU(cin, 32, batch_first=True); s.h = torch.nn.Linear(32, 6)
    def forward(s, x): hs, _ = s.g(x); return s.h(hs), hs

def train_gru(Xs, Ys, Ms, seed):
    torch.manual_seed(seed); X, valid = pad(Xs); Yt, _ = pad(Ys); M, _ = pad([m[:, None].astype(np.float32) for m in Ms])
    M = M[..., 0]*valid; net = GRUM(X.shape[2]); opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(300):
        opt.zero_grad(); out, _ = net(X); loss = (((out-Yt)**2).mean(-1)*M).sum()/M.sum()
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval(); return net

def run_gru(net, Xs):
    with torch.no_grad():
        o = [net(torch.tensor(x[None], dtype=torch.float32)) for x in Xs]
    return [a[0][0].numpy() for a in o], [a[1][0].numpy() for a in o]

def windows(seqs, W=30):  # list (T,C) -> list (T, W*C)
    out = []
    for s in seqs:
        P = np.concatenate([np.repeat(s[:1], W-1, 0), s], 0)
        out.append(np.stack([P[t:t+W].ravel() for t in range(len(s))]))
    return out

def train_mlp(X, Y, seed, H=64, ep=1000, lr=1e-3):
    rg = np.random.default_rng(seed); W1 = rg.normal(0, 1/np.sqrt(X.shape[1]), (X.shape[1], H)); b1 = np.zeros(H)
    W2 = rg.normal(0, 1/np.sqrt(H), (H, 6)); b2 = np.zeros(6); P = [W1, b1, W2, b2]; m = [np.zeros_like(p) for p in P]; v = [np.zeros_like(p) for p in P]
    for k in range(1, ep+1):
        A = np.tanh(X@W1+b1); E = (A@W2+b2-Y)*(2/Y.size)
        gW2 = A.T@E; gb2 = E.sum(0); dA = (E@W2.T)*(1-A**2); g = [X.T@dA, dA.sum(0), gW2, gb2]
        for i in range(4):
            m[i] = .9*m[i]+.1*g[i]; v[i] = .999*v[i]+.001*g[i]**2
            P[i] -= lr*(m[i]/(1-.9**k))/(np.sqrt(v[i]/(1-.999**k))+1e-8)
    return lambda Xn: np.tanh(Xn@P[0]+P[1])@P[2]+P[3]

def rm(e, m): return [float(np.sqrt(np.mean(e[j, m]**2))) for j in range(6)]

def evaluate(tau, preds, slow, dwell):
    out = {}
    for g, p in preds.items():
        e = tau-p; out[g] = {}
        for nm, m in [('overall', np.ones_like(slow)), ('motion', ~slow), ('slow', slow), ('dwell', dwell)]:
            r_ = rm(e, m); out[g][nm] = {'joints': r_, 'macro': float(np.mean(r_)), 'n': int(m.sum())}
    return out

def one_split(tr_idx, ev_split, ev_idx, seed, stage1=None):
    r, th, d = stage1 if stage1 else fit_stage1(tr_idx)
    t1_tr, z_tr = stage1_feats('train', tr_idx, r, th, d); t1_ev, z_ev = stage1_feats(ev_split, ev_idx, r, th, d)
    S_tr, S_ev = D['train'], D[ev_split]; tau_tr, tau_ev = S_tr['tau'][:, tr_idx], S_ev['tau'][:, ev_idx]
    seg_tr, seg_ev = S_tr['seg'][tr_idx], S_ev['seg'][ev_idx]
    sl_tr, _ = masks(S_tr['qd'][:, tr_idx], seg_tr); sl_ev, dw_ev = masks(S_ev['qd'][:, ev_idx], seg_ev)
    mu, sd = z_tr.mean(1, keepdims=True), z_tr.std(1, keepdims=True)+1e-8; ns = lambda a: (a-mu)/sd
    ysd = (tau_tr-t1_tr).std(1, keepdims=True)
    sq = lambda X, seg: to_seq(X, seg)
    Ztr, Zev = sq(ns(z_tr), seg_tr), sq(ns(z_ev), seg_ev)
    # stage 2
    torch.manual_seed(seed)
    f2 = train_gru(Ztr, sq((tau_tr-t1_tr)/ysd, seg_tr), [m[:, 0] > 0.5 for m in sq((~sl_tr)[None].astype(float), seg_tr)], seed)
    def s2(Zs, t1, seg):
        r2, h2 = run_gru(f2, Zs); return t1+np.concatenate(r2, 0).T*ysd, np.concatenate(h2, 0).T
    t2_tr, h_tr = s2(Ztr, t1_tr, seg_tr); t2_ev, h_ev = s2(Zev, t1_ev, seg_ev)
    tmu, tsd = t2_tr.mean(1, keepdims=True), t2_tr.std(1, keepdims=True)
    c4tr = np.concatenate([ns(z_tr), (t2_tr-tmu)/tsd, h_tr], 0); c4ev = np.concatenate([ns(z_ev), (t2_ev-tmu)/tsd, h_ev], 0)
    preds = {'G1': t1_ev, 'G3': t2_ev}
    slm_tr = [m[:, 0] > .5 for m in sq(sl_tr[None].astype(float), seg_tr)]
    for g, Ctr, Cev, base_tr, base_ev in [('G2', ns(z_tr), ns(z_ev), t1_tr, t1_ev), ('G4', c4tr, c4ev, t2_tr, t2_ev)]:
        tgt = (tau_tr-base_tr)/ysd
        f3 = train_gru(sq(Ctr, seg_tr), sq(tgt, seg_tr), slm_tr, seed)
        r3, _ = run_gru(f3, sq(Cev, seg_ev)); r3 = np.concatenate(r3, 0).T*ysd
        preds[g+'-GRU'] = base_ev + sl_ev*r3
        Wtr = np.concatenate(windows(sq(Ctr, seg_tr)), 0); Wev = np.concatenate(windows(sq(Cev, seg_ev)), 0)
        mlp = train_mlp(Wtr[sl_tr], tgt.T[sl_tr], seed)
        r3w = np.zeros_like(base_ev); r3w[:, sl_ev] = (mlp(Wev[sl_ev])).T*ysd
        preds[g+'-Win'] = base_ev + r3w
    return evaluate(tau_ev, preds, sl_ev, dw_ev), (r, th, d)

if __name__ == '__main__':
    rec = D['train']['seg']//6; folds = [[0, 1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    res = {'cv': {}, 'test': {}}; t0 = time.time()
    for fi, hold in enumerate(folds):
        tr = np.where(~np.isin(rec, hold))[0]; ev = np.where(np.isin(rec, hold))[0]; s1 = None
        for seed in F['seeds']:
            out, s1 = one_split(tr, 'train', ev, seed, s1); res['cv'][f'fold{fi}_seed{seed}'] = out
            print(f'fold{fi} seed{seed} ', {g: round(v['overall']['macro'], 4) for g, v in out.items()}, f'{time.time()-t0:.0f}s', flush=True)
    tr = np.arange(len(rec)); ev = np.arange(D['test']['tau'].shape[1]); s1 = None
    for seed in F['seeds']:
        out, s1 = one_split(tr, 'test', ev, seed, s1); res['test'][f'seed{seed}'] = out
        print('test seed', seed, {g: round(v['overall']['macro'], 4) for g, v in out.items()}, flush=True)
    res['freeze_sha256'] = hashlib.sha256(open(f'{OUT}/freeze.json', 'rb').read()).hexdigest()
    json.dump(res, open(f'{OUT}/results.json', 'w'), indent=1)

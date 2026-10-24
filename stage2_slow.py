"""Stage 2: causal temporal slow network, frozen stage 1.

Two-stage chain (user spec, 2026-10-24):
  tau_hat^(1) = tau_phy^PSD + r_motion (stage 1, FROZEN, gated fast)
  tau_hat     = tau_hat^(1) + S * r_slow(z_t)
  S = diag(0,1,1,0,1,1)  -> outputs only to joints 2,3,5,6 (idx 1,2,4,5)

Temporal state (per trajectory segment, reset at segment start, causal):
  z_{t+1} = GRU(z_t, [q, qd, qdd, m_t, d_t, h_t])
    m_t: slow flag (1 if max_j |qd_j| < 0.05, else 0)
    d_t: slow duration in seconds since last moving sample (or segment
         start), capped at 10 s, scaled by 1/10
    h_t: per-joint last significant motion direction: sign of qd_j at the
         last sample with |qd_j| > 0.1 rad/s, held otherwise, init 0
  Network state updates on EVERY sample of the segment (motion included);
  loss is computed ONLY on slow samples; slow-branch output is gated to
  zero on motion samples, so motion predictions are exactly stage 1.
  Loss: mean over slow samples of || S (e^(1) - r_slow) ||^2.

Training protocol: trajectory-block split inside train only for early
stopping -- first 10 train recordings (segments 0-59) train, last train
recording (segments 60-65) validation; test is evaluation only, never
used for selection. Adam lr 1e-3, batch 16 segments, packed sequences,
hidden 32, 1-layer GRU + linear head, seed 0.

Control (generic residual net): same architecture, but trained end-to-end
on e^(0) (PSD residual) for ALL 6 joints on ALL samples, no gating, no S,
no stage-1 freeze. This is the "everything together" baseline the
two-stage design is meant to beat on attribution (motion RMSE must not
degrade vs stage 1, slow branch must not re-absorb motion error).

Outputs: stage2_slow.json, stage2_models.pt, stage2_ablation.svg,
  stage2_slow_rmse.svg
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.manual_seed(0)
np.random.seed(0)
WORK = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(WORK, 'filtered_causal.npz'))
r0 = np.load(os.path.join(WORK, 'residuals.npz'))
s1 = np.load(os.path.join(WORK, 'stage1_e1.npz'))
S_JOINTS = [1, 2, 4, 5]  # joints 2,3,5,6
HIDDEN = 32
EPOCHS = 80
BATCH = 16
LR = 1e-3


def build_segments(split):
    q, qd, qdd, seg = z[f'{split}_q'], z[f'{split}_qd'], z[f'{split}_qdd'], z[f'{split}_seg_id']
    segs = []
    for s in np.unique(seg):
        idx = np.where(seg == s)[0]
        qs, qds, qdds = q[:, idx], qd[:, idx], qdd[:, idx]
        slow = (np.max(np.abs(qds), axis=0) < 0.05).astype(np.float32)
        age = np.zeros(len(idx), dtype=np.float32)
        last = -1
        for t in range(len(idx)):
            if slow[t] == 0:
                last = t
            age[t] = (t - last) * 0.1 if last >= 0 else t * 0.1
        age = np.minimum(age, 10.0) / 10.0
        h = np.zeros((6, len(idx)), dtype=np.float32)
        cur = np.zeros(6, dtype=np.float32)
        for t in range(len(idx)):
            for j in range(6):
                if abs(qds[j, t]) > 0.1:
                    cur[j] = np.sign(qds[j, t])
            h[:, t] = cur
        X = np.concatenate([qs.T, qds.T, qdds.T, slow[:, None], age[:, None], h.T], axis=1)
        segs.append({'X': X.astype(np.float32), 'slow': slow, 'seg': int(s)})
    return segs


def standardize(segs_fit, segs_all):
    X = np.concatenate([s['X'] for s in segs_fit], axis=0)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    for s in segs_all:
        s['X'] = (s['X'] - mu) / sd
    return mu, sd


class SlowNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, out_dim)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        out, _ = self.gru(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.head(out)


def make_batches(segs, shuffle, rng):
    order = list(range(len(segs)))
    if shuffle:
        rng.shuffle(order)
    for i in range(0, len(order), BATCH):
        batch = [segs[k] for k in order[i:i + BATCH]]
        batch = sorted(batch, key=lambda s: -len(s['slow']))
        lengths = [len(s['slow']) for s in batch]
        L = max(lengths)
        X = np.zeros((len(batch), L, batch[0]['X'].shape[1]), dtype=np.float32)
        for b, s in enumerate(batch):
            X[b, :lengths[b]] = s['X']
        yield torch.tensor(X), torch.tensor(lengths), batch, lengths


def train_model(segs_train, segs_val, target_key, out_dim, gated, slow_only):
    in_dim = segs_train[0]['X'].shape[1]
    model = SlowNet(in_dim, out_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.MSELoss(reduction='none')
    rng = np.random.RandomState(0)
    best_val, best_state, best_epoch = float('inf'), None, -1
    patience, bad = 12, 0
    for epoch in range(EPOCHS):
        model.train()
        for X, lengths, batch, lens in make_batches(segs_train, True, rng):
            opt.zero_grad()
            pred = model(X, lengths.tolist())
            loss = 0.0
            for b, s in enumerate(batch):
                tgt = torch.tensor(target_key(s)).T  # (L, out_dim)
                m = torch.tensor(s['slow']) if slow_only else torch.ones_like(torch.tensor(s['slow']))
                if gated:
                    pred_b = pred[b, :lens[b]] * torch.tensor(s['slow'])[:, None]
                else:
                    pred_b = pred[b, :lens[b]]
                sq = ((tgt[:lens[b]] - pred_b) ** 2).sum(dim=1)
                loss = loss + (sq * m[:lens[b]]).sum() / max(m[:lens[b]].sum().item(), 1)
            loss = loss / len(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # validation: slow RMSE on S joints (or all joints for generic)
        val_rmse = evaluate(model, segs_val, target_key, out_dim, gated, slow_only, regime='slow')
        if val_rmse < best_val - 1e-4:
            best_val, best_state, best_epoch = val_rmse, {k: v.clone() for k, v in model.state_dict().items()}, epoch
            bad = 0
        else:
            bad += 1
        if epoch % 10 == 0 or bad >= patience:
            print(f'  epoch {epoch}: val slow RMSE {val_rmse:.4f} (best {best_val:.4f} @ {best_epoch})')
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, {'best_epoch': best_epoch, 'best_val_slow_rmse': best_val}


def predict(model, segs, out_dim, gated):
    model.eval()
    preds = []
    with torch.no_grad():
        for s in segs:
            X = torch.tensor(s['X'][None])
            out = model(X, [len(s['slow'])])[0, :, :].numpy()
            if gated:
                out = out * s['slow'][:, None]
            preds.append(out.T)  # (out_dim, L)
    return preds


def evaluate(model, segs, target_key, out_dim, gated, slow_only, regime='overall'):
    preds = predict(model, segs, out_dim, gated)
    sq_all = []
    for s, p in zip(segs, preds):
        tgt = target_key(s)
        slow = s['slow'] > 0.5
        if regime == 'slow':
            m = slow
        elif regime == 'motion':
            m = ~slow
        else:
            m = np.ones_like(slow)
        sq_all.append(((tgt[:, m] - p[:, m]) ** 2).mean())
    return float(np.sqrt(np.mean(sq_all)))


train_segs = build_segments('train')
test_segs = build_segments('test')
# attach targets
for s in train_segs:
    idx = np.where(z['train_seg_id'] == s['seg'])[0]
    s['e1'] = s1['train_delivered'][:, idx][S_JOINTS]
    s['e0'] = r0['train_resid'][:, idx]
for s in test_segs:
    idx = np.where(z['test_seg_id'] == s['seg'])[0]
    s['e1'] = s1['test_delivered'][:, idx][S_JOINTS]
    s['e0'] = r0['test_resid'][:, idx]

mu, sd = standardize(train_segs, train_segs + test_segs)
val_segs = [s for s in train_segs if s['seg'] >= 60]
fit_segs = [s for s in train_segs if s['seg'] < 60]
print(f'train fit segs {len(fit_segs)}, val segs {len(val_segs)}, test segs {len(test_segs)}')

print('== stage 2: slow GRU (frozen stage 1, slow-only loss, gated) ==')
slow_model, slow_info = train_model(fit_segs, val_segs, lambda s: s['e1'], len(S_JOINTS), gated=True, slow_only=True)
print('slow model info:', slow_info)
print('== control: generic residual GRU (e0, all joints, all samples, no gating) ==')
gen_model, gen_info = train_model(fit_segs, val_segs, lambda s: s['e0'], 6, gated=False, slow_only=False)
print('generic model info:', gen_info)

torch.save({'slow': slow_model.state_dict(), 'generic': gen_model.state_dict(),
            'mu': mu, 'sd': sd, 'hidden': HIDDEN, 'S_joints': S_JOINTS},
           os.path.join(WORK, 'stage2_models.pt'))


def full_metrics(pred_resid_fn):
    out = {}
    for split, segs in [('train', train_segs), ('test', test_segs)]:
        resid = pred_resid_fn(segs)
        slow_all = np.concatenate([s['slow'] for s in segs])
        out[split] = {}
        for regime, m in [('overall', np.ones_like(slow_all, dtype=bool)),
                          ('motion', slow_all < 0.5), ('slow', slow_all > 0.5)]:
            per = np.sqrt(np.mean(resid[:, m] ** 2, axis=1))
            out[split][regime] = {'mean': float(per.mean()), 'per_joint': per.tolist(),
                                  'S_mean': float(per[S_JOINTS].mean()), 'J1': float(per[0])}
    return out


def resid_stage1(segs):
    return np.concatenate([s1['train_delivered'][:, np.where(z['train_seg_id'] == s['seg'])[0]]
                           if s in train_segs else s1['test_delivered'][:, np.where(z['test_seg_id'] == s['seg'])[0]]
                           for s in segs], axis=1) if False else None


# Build residuals directly per split (avoid the fragile helper above)
def build_resid(split, segs, kind):
    e1 = s1[f'{split}_delivered']
    e0 = r0[f'{split}_resid']
    seg_ids = z[f'{split}_seg_id']
    cols = []
    for s in segs:
        cols.append(np.where(seg_ids == s['seg'])[0])
    idx = np.concatenate(cols)
    if kind == 'phy':
        return e0[:, idx]
    if kind == 'stage1':
        return e1[:, idx]
    if kind == 'stage2':
        preds = predict(slow_model, segs, len(S_JOINTS), gated=True)
        r = e1[:, idx].copy()
        pos = 0
        for s, p in zip(segs, preds):
            L = len(s['slow'])
            r[S_JOINTS, pos:pos + L] = e1[:, idx][S_JOINTS, pos:pos + L] - p
            pos += L
        return r
    if kind == 'generic':
        preds = predict(gen_model, segs, 6, gated=False)
        r = e0[:, idx].copy()
        pos = 0
        for s, p in zip(segs, preds):
            L = len(s['slow'])
            r[:, pos:pos + L] = e0[:, idx][:, pos:pos + L] - p
            pos += L
        return r


metrics = {}
for kind in ['phy', 'stage1', 'stage2', 'generic']:
    metrics[kind] = {}
    for split, segs in [('train', train_segs), ('test', test_segs)]:
        resid = build_resid(split, segs, kind)
        slow_all = np.concatenate([s['slow'] for s in segs])
        metrics[kind][split] = {}
        for regime, m in [('overall', np.ones_like(slow_all, dtype=bool)),
                          ('motion', slow_all < 0.5), ('slow', slow_all > 0.5)]:
            per = np.sqrt(np.mean(resid[:, m] ** 2, axis=1))
            metrics[kind][split][regime] = {'mean': float(per.mean()), 'per_joint': per.tolist(),
                                            'S_mean': float(per[S_JOINTS].mean()), 'J1': float(per[0])}
        print(f"{kind} {split}: overall {metrics[kind][split]['overall']['mean']:.3f} "
              f"motion {metrics[kind][split]['motion']['mean']:.3f} "
              f"slow {metrics[kind][split]['slow']['mean']:.3f} | "
              f"slow per-joint {np.round(metrics[kind][split]['slow']['per_joint'], 3)}")

json.dump({'slow_info': slow_info, 'generic_info': gen_info, 'metrics': metrics,
           'note': ('Stage 1 frozen. Slow GRU: state updates on all samples, loss only on '
                    'slow samples, output gated to zero on motion samples, state reset per '
                    'segment, early stopping on last train recording (segments 60-65). '
                    'Generic control: same GRU trained end-to-end on e0, all joints, all samples.')},
          open(os.path.join(WORK, 'stage2_slow.json'), 'w'), indent=1)
print('saved stage2_slow.json, stage2_models.pt')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, regime in zip(axes, ['overall', 'motion', 'slow']):
    x = np.arange(2)
    w = 0.2
    for k, name, col in [('phy', 'phy', 'C0'), ('stage1', 'phy+motion', 'C1'),
                         ('stage2', 'phy+motion+slow', 'C2'), ('generic', 'generic net', 'C3')]:
        vals = [metrics[k][s][regime]['mean'] for s in ['train', 'test']]
        ax.bar(x + (['phy', 'stage1', 'stage2', 'generic'].index(k) - 1.5) * w, vals, w, label=name, color=col)
    ax.set_xticks(x); ax.set_xticklabels(['train', 'test']); ax.set_ylabel('mean RMSE (Nm)')
    ax.set_title(regime); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
fig.suptitle('Two-stage ablation: mean RMSE by regime')
fig.tight_layout(); fig.savefig(os.path.join(WORK, 'stage2_ablation.svg')); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(6); w = 0.35
ax.bar(x - w / 2, metrics['stage1']['test']['slow']['per_joint'], w, label='phy+motion (test, slow)', color='C1')
ax.bar(x + w / 2, metrics['stage2']['test']['slow']['per_joint'], w, label='phy+motion+slow (test, slow)', color='C2')
ax.set_xticks(x); ax.set_xticklabels([f'J{j+1}' for j in range(6)])
ax.set_ylabel('slow-sample RMSE (Nm)'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
ax.set_title('Test slow-sample RMSE per joint: stage 1 vs stage 1 + slow GRU (J1/J4 have no slow output by design)')
fig.tight_layout(); fig.savefig(os.path.join(WORK, 'stage2_slow_rmse.svg')); plt.close(fig)
print('saved stage2_ablation.svg, stage2_slow_rmse.svg')

"""
Train the Pillar-PointNet segmentation network (pure NumPy, runs anywhere, CPU only).

Architecture (PointNet-style shared MLP + two-scale pillar max-pooling for context)
    per point  x(12) -> 32 -> 32  = a          (shared MLP)
    pool       p1 = max of a over the 0.4 m pillar, p2 = max over the 1.6 m pillar
    head       [a, p1, p2](96) -> 64 -> 8 classes

The 12 input features are produced by the C++ core (tf_features), so training and the real-time
C++ inference see bit-identical inputs. Half of the steps pool over all points and half over the
"uncertain" subset only, so the same weights work with and without selective AI.

    python python/train.py --data data --epochs 24
For large real datasets (SemanticKITTI / RELLIS-3D) use train_torch.py on a GPU; it writes the same file.
"""
import argparse
import glob
import os
import time
import numpy as np
from engine import Engine, Sequence, LUT, ROOT, CLASS_NAMES, NFEAT

H1, H2, H3, NC = 32, 32, 64, 8


def load_frames(dirs, every=1):
    eng = Engine('policy=1;selective_ai=1', model=None)
    out = []
    for d in dirs:
        seq = Sequence(d)
        for i in range(0, len(seq), every):
            lab = seq.labels(i)
            f, g1, g2, tri = eng.features(seq.scan(i), seq.poses[i])
            y = LUT[lab]
            m = (tri != 255) & (y > 0)
            out.append((f[m], g1[m], g2[m], tri[m], y[m]))
    return out


def pool(a, g):
    order = np.argsort(g, kind='stable'); gs = g[order]
    starts = np.r_[0, np.nonzero(gs[1:] != gs[:-1])[0] + 1]
    inv = np.empty(len(g), np.int64); inv[order] = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(g)]))
    return np.maximum.reduceat(a[order], starts, axis=0), inv, order, starts


def forward(P, x, g1, g2):
    xn = (x - P['mu']) / P['sd']
    a1 = np.maximum(xn @ P['W1'].T + P['b1'], 0)
    a2 = np.maximum(a1 @ P['W2'].T + P['b2'], 0)
    p1, i1, o1, s1 = pool(a2, g1); p2, i2, o2, s2 = pool(a2, g2)
    z = np.concatenate([a2, p1[i1], p2[i2]], 1)
    a3 = np.maximum(z @ P['W3'].T + P['b3'], 0)
    lg = a3 @ P['W4'].T + P['b4']
    return lg, (xn, a1, a2, p1, i1, o1, s1, p2, i2, o2, s2, z, a3)


def backward(P, cache, dlg):
    xn, a1, a2, p1, i1, o1, s1, p2, i2, o2, s2, z, a3 = cache
    G = {'W4': dlg.T @ a3, 'b4': dlg.sum(0)}
    da3 = (dlg @ P['W4']) * (a3 > 0)
    G['W3'] = da3.T @ z; G['b3'] = da3.sum(0)
    dz = da3 @ P['W3']
    da2 = dz[:, :H2].copy()
    for k, (p, inv, order, starts) in enumerate(((p1, i1, o1, s1), (p2, i2, o2, s2))):
        d = dz[:, H2 * (k + 1):H2 * (k + 2)]
        dp = np.add.reduceat(d[order], starts, axis=0)            # gradient arriving at each pooled cell
        da2 += (a2 == p[inv]) * dp[inv]                            # routed to the arg-max point
    da2 *= (a2 > 0)
    G['W2'] = da2.T @ a1; G['b2'] = da2.sum(0)
    da1 = (da2 @ P['W2']) * (a1 > 0)
    G['W1'] = da1.T @ xn; G['b1'] = da1.sum(0)
    return G


def iou_table(conf):
    tp = np.diag(conf).astype(float); den = conf.sum(0) + conf.sum(1) - tp
    return np.where(den > 0, tp / np.maximum(den, 1), np.nan)


def evaluate(P, frames, subset=False):
    conf = np.zeros((NC, NC), np.int64)
    for f, g1, g2, tri, y in frames:
        if subset:
            m = tri == 1; pred = np.where(tri == 0, 1, 2).astype(np.int64)
            if m.any():
                lg, _ = forward(P, f[m], g1[m], g2[m]); pred[m] = lg.argmax(1)
        else:
            lg, _ = forward(P, f, g1, g2); pred = lg.argmax(1)
        np.add.at(conf, (y, pred), 1)
    return iou_table(conf), conf


def save_bin(P, path):
    with open(path, 'wb') as fp:
        fp.write(b'TFNET01\0'); np.array([NFEAT, H1, H2, H3, NC], np.int32).tofile(fp)
        for k in ('mu', 'sd', 'W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'):
            np.ascontiguousarray(P[k], np.float32).tofile(fp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(ROOT, 'data'))
    ap.add_argument('--train', nargs='*', help='sequence folders to train on (default: <data>/train_*)')
    ap.add_argument('--val', nargs='*', help='sequence folders to validate on (default: <data>/val_*)')
    ap.add_argument('--every', type=int, default=1, help='use every n-th sweep (real sequences are long)')
    ap.add_argument('--epochs', type=int, default=24)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'pillar_pointnet.bin'))
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    tr = load_frames(a.train or sorted(glob.glob(os.path.join(a.data, 'train_*'))), every=a.every)
    va = load_frames(a.val or sorted(glob.glob(os.path.join(a.data, 'val_*'))), every=6 * a.every)
    print(f'{len(tr)} training sweeps ({sum(len(f[0]) for f in tr):,} points), {len(va)} validation sweeps')
    allx = np.concatenate([f[0] for f in tr]); ally = np.concatenate([f[4] for f in tr])
    cnt = np.bincount(ally, minlength=NC).astype(float)
    cw = np.where(cnt > 0, (cnt.sum() / np.maximum(cnt, 1)) ** 0.5, 0); cw = (cw / cw[cnt > 0].mean()).astype(np.float32)
    P = {'mu': allx.mean(0), 'sd': allx.std(0) + 1e-3}

    def he(o, i):
        return (rng.standard_normal((o, i)) * np.sqrt(2.0 / i)).astype(np.float32)
    P.update(W1=he(H1, NFEAT), b1=np.zeros(H1, np.float32), W2=he(H2, H1), b2=np.zeros(H2, np.float32),
             W3=he(H3, 3 * H2), b3=np.zeros(H3, np.float32), W4=he(NC, H3), b4=np.zeros(NC, np.float32))
    keys = [k for k in P if k[0] in 'Wb']
    M = {k: np.zeros_like(P[k]) for k in keys}; V = {k: np.zeros_like(P[k]) for k in keys}; step = 0
    t0 = time.time()
    for ep in range(a.epochs):
        lr = a.lr * 0.5 * (1 + np.cos(np.pi * ep / a.epochs)); tot = 0.0
        for j in rng.permutation(len(tr)):
            f, g1, g2, tri, y = tr[j]
            if rng.random() < 0.5:                                  # selective-AI mode: uncertain points only
                m = tri == 1; f, g1, g2, y = f[m], g1[m], g2[m], y[m]
            if len(y) < 100:
                continue
            f = f + rng.normal(0, 0.01, f.shape).astype(np.float32) * P['sd']
            lg, cache = forward(P, f, g1, g2)
            lg = lg - lg.max(1, keepdims=True); p = np.exp(lg); p /= p.sum(1, keepdims=True)
            w = cw[y]; n = len(y)
            tot += float(-(w * np.log(p[np.arange(n), y] + 1e-9)).sum() / w.sum())
            d = p; d[np.arange(n), y] -= 1; d *= (w / w.sum())[:, None]
            G = backward(P, cache, d.astype(np.float32)); step += 1
            for k in keys:
                M[k] = 0.9 * M[k] + 0.1 * G[k]; V[k] = 0.999 * V[k] + 0.001 * G[k] ** 2
                P[k] -= (lr * (M[k] / (1 - 0.9 ** step)) / (np.sqrt(V[k] / (1 - 0.999 ** step)) + 1e-8)).astype(np.float32)
        if ep % 4 == 3 or ep == a.epochs - 1:
            iou, _ = evaluate(P, va)
            print(f'epoch {ep + 1:3d}  loss {tot / len(tr):.4f}  val mIoU {np.nanmean(iou[1:]):.3f}  ({time.time() - t0:.0f}s)', flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save_bin(P, a.out); np.savez(a.out.replace('.bin', '.npz'), **P)
    for name, sub in (('all points through the network', False), ('selective AI (triage + network)', True)):
        iou, _ = evaluate(P, va, sub)
        print(f'\nvalidation, {name}: mIoU {np.nanmean(iou[1:]):.3f}')
        print('  ' + '  '.join(f'{CLASS_NAMES[k]} {iou[k]:.3f}' for k in range(1, NC)))
    print('\nsaved', a.out)

"""PillarPointNet: the learned stage.

A PointNet++-style segmentation network in which the neighbourhoods are
vertical pillars at three scales (0.4 m, 1.6 m, 4.8 m) instead of ball queries:

    per-point MLP  ->  max-pool inside each pillar, at every scale (set abstraction)
                   ->  concatenate [point feature | 3 pooled context vectors | pillar statistics]
                   ->  MLP  ->  8-class logits per point

Grid grouping costs O(N); ball-query grouping in PointNet++ costs O(N log N) with
a large constant and is the reason that model is slow on 100k-point sweeps.

This file is the complete model in NumPy - forward pass, analytic backward pass
and Adam - so it trains and runs on any laptop with no deep-learning framework.
`scripts/train_torch.py` holds the same architecture in PyTorch for GPU training on
SemanticKITTI / RELLIS-3D; both read and write the same .npz weight file.
"""
from __future__ import annotations

import numpy as np

from . import classes as C

F = np.float32
SCALES = (0.4, 1.6, 4.8)
D_IN = 3 + 3 * len(SCALES)
D_STAT = 4 * len(SCALES)
H1, H2, H3, H4 = 48, 48, 96, 64
D_CAT = H2 * (1 + len(SCALES)) + D_STAT


def _segments(key):
    order = np.argsort(key, kind="stable")
    s = key[order]
    starts = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    seg = np.empty(len(key), np.int64)
    seg[order] = np.cumsum(np.concatenate(([0], (s[1:] != s[:-1]).astype(np.int64))))
    return order, starts, seg


VOXEL = 0.15


def voxel_representatives(x, y, z, size=VOXEL):
    """One representative point per occupied voxel, plus the map from every point to its voxel.
    The network labels the representatives; all points in a voxel share that label."""
    ix = np.floor(x / size).astype(np.int64); iy = np.floor(y / size).astype(np.int64)
    iz = np.floor(z / size).astype(np.int64)
    ix -= ix.min(); iy -= iy.min(); iz -= iz.min()
    key = (ix * (int(iy.max()) + 1) + iy) * (int(iz.max()) + 1) + iz
    _, rep, inv = np.unique(key, return_index=True, return_inverse=True)
    return rep, inv


def segmax(a_sorted, starts):
    """Max over consecutive row segments, as a tournament: neighbouring rows of the same segment
    are paired and reduced, halving the row count each round.  O(N) work in log2(longest segment)
    vectorised rounds; np.maximum.reduceat along axis 0 is ~10x slower on this shape of data."""
    n = len(a_sorted)
    vals = a_sorted
    seg = np.zeros(n, np.int64)
    seg[starts[1:]] = 1
    seg = np.cumsum(seg)
    nseg = len(starts)
    while len(vals) > nseg:
        first = np.concatenate(([True], seg[1:] != seg[:-1]))
        pos = np.arange(len(seg)) - np.maximum.accumulate(np.where(first, np.arange(len(seg)), 0))
        even = np.flatnonzero((pos & 1) == 0)
        nxt = np.minimum(even + 1, len(seg) - 1)
        has = (even + 1 < len(seg)) & (seg[nxt] == seg[even])
        out = vals[even]
        j = np.flatnonzero(has)
        out[j] = np.maximum(out[j], vals[even[j] + 1])
        vals, seg = out, seg[even]
    return vals


def build_features(x, y, z, h, intensity, rng, scales=SCALES):
    """Per-point input features, per-point pillar statistics, and pillar grouping for every scale."""
    n = len(x)
    f0 = np.empty((n, D_IN), F)
    f0[:, 0] = np.clip(h, -1.0, 6.0) * 0.4
    f0[:, 1] = intensity
    f0[:, 2] = rng * 0.02
    stats = np.empty((n, D_STAT), F)
    groups = []
    for k, s in enumerate(scales):
        ix = np.floor(x / s).astype(np.int64); iy = np.floor(y / s).astype(np.int64)
        key = (ix - ix.min()) * (int(iy.max() - iy.min()) + 1) + (iy - iy.min())
        order, starts, seg = _segments(key)
        cnt = np.diff(np.concatenate((starts, [n]))).astype(F)
        mz = np.add.reduceat(z[order].astype(np.float64), starts) / cnt
        hmax = np.maximum.reduceat(h[order], starts); hmin = np.minimum.reduceat(h[order], starts)
        f0[:, 3 + 3 * k] = (x / s - ix) - 0.5
        f0[:, 4 + 3 * k] = (y / s - iy) - 0.5
        f0[:, 5 + 3 * k] = np.clip(z - mz[seg], -3, 3)
        stats[:, 4 * k] = np.log1p(cnt[seg]) * 0.2
        stats[:, 4 * k + 1] = np.clip(hmax[seg] - hmin[seg], 0, 6) * 0.3
        stats[:, 4 * k + 2] = np.clip(hmin[seg], -1, 6) * 0.4
        stats[:, 4 * k + 3] = np.clip(hmax[seg], -1, 6) * 0.3
        groups.append((order, starts, seg))
    return f0, stats, groups


def init_params(seed=0):
    r = np.random.default_rng(seed)
    def lin(i, o):
        return (r.standard_normal((i, o)) * np.sqrt(2.0 / i)).astype(F), np.zeros(o, F)
    p = {}
    for name, (i, o) in {"1": (D_IN, H1), "2": (H1, H2), "3": (D_CAT, H3), "4": (H3, H4), "5": (H4, C.NUM_CLASSES)}.items():
        p["W" + name], p["b" + name] = lin(i, o)
    return p


def forward(p, f0, stats, groups, cache=False):
    a1 = np.maximum(f0 @ p["W1"] + p["b1"], 0)
    a2 = np.maximum(a1 @ p["W2"] + p["b2"], 0)
    parts = [a2]
    for order, starts, seg in groups:
        parts.append(segmax(a2[order], starts)[seg])
    parts.append(stats)
    cat = np.concatenate(parts, axis=1)
    a3 = np.maximum(cat @ p["W3"] + p["b3"], 0)
    a4 = np.maximum(a3 @ p["W4"] + p["b4"], 0)
    logits = a4 @ p["W5"] + p["b5"]
    if cache:
        return logits, (f0, a1, a2, parts, cat, a3, a4)
    return logits


def softmax(logits):
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def loss_and_grads(p, f0, stats, groups, labels, class_w):
    """Weighted cross-entropy and its exact gradient with respect to every parameter."""
    logits, (f0, a1, a2, parts, cat, a3, a4) = forward(p, f0, stats, groups, cache=True)
    prob = softmax(logits.astype(np.float64))
    n = len(labels)
    w = class_w[labels]
    wsum = w.sum()
    loss = float(-(w * np.log(prob[np.arange(n), labels] + 1e-12)).sum() / wsum)
    d = prob
    d[np.arange(n), labels] -= 1.0
    d = (d * (w / wsum)[:, None]).astype(F)
    g = {}
    g["W5"] = a4.T @ d; g["b5"] = d.sum(0)
    d = (d @ p["W5"].T) * (a4 > 0)
    g["W4"] = a3.T @ d; g["b4"] = d.sum(0)
    d = (d @ p["W4"].T) * (a3 > 0)
    g["W3"] = cat.T @ d; g["b3"] = d.sum(0)
    dcat = d @ p["W3"].T
    d2 = dcat[:, :H2].copy()
    for k, (order, starts, seg) in enumerate(groups):
        gp = dcat[:, H2 * (k + 1):H2 * (k + 2)]
        # each pooled value came from the point(s) that hold the maximum of that channel in the pillar
        gsum = np.add.reduceat(gp[order], starts, axis=0)[seg]
        is_max = a2 == parts[k + 1]
        ties = np.add.reduceat(is_max[order].astype(F), starts, axis=0)[seg]
        d2 += is_max * gsum / np.maximum(ties, 1)
    d2 *= (a2 > 0)
    g["W2"] = a1.T @ d2; g["b2"] = d2.sum(0)
    d1 = (d2 @ p["W2"].T) * (a1 > 0)
    g["W1"] = f0.T @ d1; g["b1"] = d1.sum(0)
    return loss, g, logits


class Adam:
    def __init__(self, params, lr=2e-3, b1=0.9, b2=0.999, wd=1e-5):
        self.lr, self.b1, self.b2, self.wd, self.t = lr, b1, b2, wd, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        for k in params:
            g = grads[k].astype(F) + (self.wd * params[k] if k[0] == "W" else 0)
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t); vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(F)


class PillarPointNet:
    """Inference wrapper used by the pipeline."""

    def __init__(self, params):
        self.p = {k: np.ascontiguousarray(v, dtype=F) for k, v in params.items() if k[0] in "Wb"}
        for v in self.p.values():
            # weight decay drives dead units to denormal floats (~1e-40); one such value in a
            # matrix makes the CPU fall back to microcode and the layer runs ~10x slower.
            v[np.abs(v) < 1e-20] = 0.0

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def save(self, path, **meta):
        np.savez(path, **self.p, **meta)

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def predict(self, x, y, z, h, intensity, rng):
        """Returns (class id, confidence) per point."""
        if len(x) == 0:
            return np.zeros(0, np.int8), np.zeros(0, F)
        rep, inv = voxel_representatives(x, y, z)
        f0, stats, groups = build_features(x[rep], y[rep], z[rep], h[rep], intensity[rep], rng[rep])
        prob = softmax(forward(self.p, f0, stats, groups))
        self.last_n = len(rep)
        return prob.argmax(1).astype(np.int8)[inv], prob.max(1).astype(F)[inv]

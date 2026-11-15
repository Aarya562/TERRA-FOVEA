"""Dynamic objects: cluster -> associate -> constant-velocity Kalman filter.

Only returns that the network calls vehicle or pedestrian are clustered, so a
wall never becomes a track.  A track is 'moving' once its filtered speed stays
above a threshold; parked cars and standing people stay static.  The predicted
positions of moving tracks feed the resolution policy.
"""
import numpy as np
from scipy.ndimage import label as cc_label, binary_dilation
from scipy.optimize import linear_sum_assignment

from .config import Config
from . import classes as C


def cluster(x, y, cls, ego_xy, cfg: Config, half=92.0):
    """Connected components on a BEV raster.  Returns per-point cluster id (-1 = none) and cluster table."""
    res = cfg.cluster_res
    n = int(2 * half / res)
    ix = ((x - ego_xy[0] + half) / res).astype(np.int64); iy = ((y - ego_xy[1] + half) / res).astype(np.int64)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    img = np.zeros((n, n), bool)
    img[ix[ok], iy[ok]] = True
    lab, k = cc_label(binary_dilation(img, np.ones((3, 3), bool)))
    pid = np.full(x.shape, -1, np.int64)
    pid[ok] = lab[ix[ok], iy[ok]] - 1
    dets = []
    if k == 0:
        return pid, dets
    v = pid >= 0
    cnt = np.bincount(pid[v], minlength=k)
    sx = np.bincount(pid[v], weights=x[v], minlength=k); sy = np.bincount(pid[v], weights=y[v], minlength=k)
    xmin = np.full(k, np.inf); xmax = np.full(k, -np.inf); ymin = np.full(k, np.inf); ymax = np.full(k, -np.inf)
    np.minimum.at(xmin, pid[v], x[v]); np.maximum.at(xmax, pid[v], x[v])
    np.minimum.at(ymin, pid[v], y[v]); np.maximum.at(ymax, pid[v], y[v])
    nped = np.bincount(pid[v], weights=(cls[v] == C.PEDESTRIAN), minlength=k)
    for i in range(k):
        ex, ey = xmax[i] - xmin[i], ymax[i] - ymin[i]
        if cnt[i] < 4 or max(ex, ey) > 12.0:
            continue
        c = C.PEDESTRIAN if nped[i] * 2 > cnt[i] else C.VEHICLE
        if c == C.PEDESTRIAN and max(ex, ey) > 1.8:
            c = C.VEHICLE
        dets.append({"cid": i, "x": 0.5 * (xmin[i] + xmax[i]) if c == C.VEHICLE else sx[i] / cnt[i],
                     "y": 0.5 * (ymin[i] + ymax[i]) if c == C.VEHICLE else sy[i] / cnt[i],
                     "mx": sx[i] / cnt[i], "my": sy[i] / cnt[i],
                     "ex": ex, "ey": ey, "n": int(cnt[i]), "cls": c,
                     "box": (xmin[i], xmax[i], ymin[i], ymax[i])})
    return pid, dets


def _edge_speed(boxes):
    """Speed that BOTH opposite edges of the footprint agree on.

    When the ego drives past a parked car, more of the car comes into view: one edge
    of its visible footprint stays put while the other grows, and the centroid drifts
    as if the car were moving.  A really moving object drags both edges along.
    """
    if len(boxes) < 5:
        return 0.0
    a, b = boxes[0], boxes[-1]
    T = max(b[0] - a[0], 1e-3)
    v = []
    for lo, hi in ((1, 2), (3, 4)):
        d0, d1 = b[lo] - a[lo], b[hi] - a[hi]
        v.append(0.0 if d0 * d1 <= 0 else np.sign(d0) * min(abs(d0), abs(d1)) / T)
    return float(np.hypot(v[0], v[1]))


class Tracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracks = []
        self.next_id = 1

    def step(self, dets, dt):
        cfg = self.cfg
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        q = 3.0
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        Q = G @ G.T * q * q
        H = np.zeros((2, 4)); H[0, 0] = H[1, 1] = 1
        for t in self.tracks:
            t["s"] = F @ t["s"]; t["P"] = F @ t["P"] @ F.T + Q
        assigned = {}
        if self.tracks and dets:
            tp = np.array([t["s"][:2] for t in self.tracks]); dp = np.array([[d["mx"], d["my"]] for d in dets])
            cost = np.linalg.norm(tp[:, None] - dp[None], axis=2)
            gate = np.array([cfg.track_gate if t["cls"] == C.VEHICLE else 1.2 for t in self.tracks])[:, None]
            cost_g = np.where(cost < gate, cost, 1e6)
            ri, ci = linear_sum_assignment(cost_g)
            for a, b in zip(ri, ci):
                if cost_g[a, b] < 1e5:
                    assigned[a] = b
        used = set(assigned.values())
        for i, t in enumerate(self.tracks):
            if i in assigned:
                d = dets[assigned[i]]
                R = np.eye(2) * (0.35 if t["cls"] == C.VEHICLE else 0.12) ** 2
                z = np.array([d["mx"], d["my"]])
                S = H @ t["P"] @ H.T + R
                Kg = t["P"] @ H.T @ np.linalg.inv(S)
                t["s"] = t["s"] + Kg @ (z - H @ t["s"])
                t["P"] = (np.eye(4) - Kg @ H) @ t["P"]
                t["hits"] += 1; t["miss"] = 0; t["cid"] = d["cid"]
                t["age"] += dt
                t["path"].append((float(t["s"][0]), float(t["s"][1])))
                if len(t["path"]) > 60:
                    t["path"].pop(0)
                t["boxes"].append((t["age"],) + tuple(d["box"]))
                if len(t["boxes"]) > 9:
                    t["boxes"].pop(0)
                t["ex"], t["ey"], t["n"] = d["ex"], d["ey"], d["n"]
                if d["cls"] == t["cls"]:
                    t["cls_votes"] += 1
                else:
                    t["cls_votes"] -= 1
                    if t["cls_votes"] < 0:
                        t["cls"], t["cls_votes"] = d["cls"], 0
            else:
                t["miss"] += 1; t["cid"] = -1; t["age"] += dt
            sp = float(np.hypot(t["s"][2], t["s"][3]))
            t["edge_speed"] = _edge_speed(t["boxes"])
            if t["hits"] >= cfg.track_confirm and sp > cfg.moving_speed and t["edge_speed"] > 0.6 * cfg.moving_speed:
                t["mov_count"] = min(t["mov_count"] + 1, 10)
            else:
                t["mov_count"] = max(t["mov_count"] - 1, 0)
        for j, d in enumerate(dets):
            if j in used:
                continue
            P0 = np.diag([0.3, 0.3, 25.0, 25.0])
            self.tracks.append({"id": self.next_id, "s": np.array([d["mx"], d["my"], 0.0, 0.0]), "P": P0,
                                "cls": d["cls"], "cls_votes": 0, "hits": 1, "miss": 0, "cid": d["cid"],
                                "ex": d["ex"], "ey": d["ey"], "n": d["n"], "mov_count": 0,
                                "age": 0.0, "path": [(float(d["mx"]), float(d["my"]))], "boxes": [(0.0,) + tuple(d["box"])], "edge_speed": 0.0})
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= cfg.track_max_miss]
        return self.public()

    def public(self):
        out = []
        for t in self.tracks:
            out.append({"id": t["id"], "cls": t["cls"], "x": float(t["s"][0]), "y": float(t["s"][1]),
                        "vx": float(t["s"][2]), "vy": float(t["s"][3]),
                        "speed": float(np.hypot(t["s"][2], t["s"][3])),
                        "confirmed": t["hits"] >= self.cfg.track_confirm, "moving": t["mov_count"] >= 3,
                        "cid": t["cid"], "ex": float(t["ex"]), "ey": float(t["ey"]), "n": t["n"],
                        "visible": t["miss"] == 0, "path": list(t["path"])})
        return out

"""Run the C++ core (core/terra_fovea.cpp) and present its map to the dashboard.

The C++ engine is the real-time implementation; this file only translates its exported
cells, tracks, statistics and driving decision into the dashboard's frame format
(terrafovea/export.py).  If the shared library has not been built, importing this raises
SystemExit and the caller falls back to the NumPy reference pipeline.
"""
import json
import os
import struct
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
import engine as E                      # noqa: E402  (ctypes binding written next to the core)
from .export import REC                 # noqa: E402

# C++ class ids -> dashboard class ids (0 unknown, 1 drivable, 2 terrain, 3 structure, 4 pole, 5 vegetation, 6 vehicle, 7 person)
_CLS = np.array([255, 0, 1, 2, 3, 4, 5, 6], np.uint8)
POLICY_CFG = {"risk": "policy=2", "distance": "policy=1", "uniform_coarse": "policy=0;uniform_level=3",
              "uniform_fine": "policy=0;uniform_level=0"}
TILE = 1.6


def make_engine(policy="risk", ray_clearing=True):
    return E.Engine(POLICY_CFG[policy] + (";ray_clear=1" if ray_clearing else ";ray_clear=0"))


def baselines():
    """Reference memory figures measured by python/evaluate.py."""
    p = os.path.join(ROOT, "results", "metrics.json")
    if not os.path.exists(p):
        return {}
    R = json.load(open(p))["results"]
    out = {"dense_2p5d_5cm_mb": 4000 * 4000 * 16 / 1e6, "dense_3d_voxel_5cm_mb": 4000 * 4000 * 80 / 1e6}
    if "uniform_5cm" in R:
        out["uniform_5cm_sparse_mb"] = R["uniform_5cm"]["mean"]["map_bytes"] / 1e6
    return out


def _one_per_key(kx, ky):
    """Index of one element per distinct (kx, ky), via a dense lookup (much faster than np.unique here)."""
    if len(kx) == 0:
        return np.zeros(0, np.int64)
    kx = kx - kx.min(); ky = ky - ky.min()
    w = int(ky.max()) + 1
    slot = np.full((int(kx.max()) + 1) * w, -1, np.int64)
    slot[kx * w + ky] = np.arange(len(kx))
    return slot[slot >= 0]


def build_frame(eng, seq_no, t, policy, ray_clearing, view_radius=100.0, extra=None, ground_lod=0.2) -> bytes:
    c = eng.cells()        # x y size ground top clearance class conf occ flags neg level risk
    s = eng.stats()
    d = eng.decision()
    ex, ey, yaw = d.get("x", 0.0), d.get("y", 0.0), d.get("yaw", 0.0)
    ox = float(np.round(ex / TILE) * TILE); oy = float(np.round(ey / TILE) * TILE)
    c = c[np.hypot(c[:, 0] - ex, c[:, 1] - ey) <= view_radius]
    near = np.hypot(c[:, 0] - ex, c[:, 1] - ey) < 5.0
    z0 = float(np.median(c[near, 3])) if near.any() else 0.0
    flags = c[:, 9].astype(np.int64)
    dyn = (flags & E.C_DYNAMIC) > 0
    occ = (c[:, 4] >= 0.10) & ((c[:, 8] >= 0.5) | dyn)
    hole = (c[:, 10] >= 0.5) | ((flags & E.C_NEGATIVE) > 0)
    over = (c[:, 5] >= 0) & ~occ
    length, hw = d.get("corridor_len", 30.0), d.get("corridor_hw", 1.2)
    cs_, sn_ = np.cos(yaw), np.sin(yaw)
    dx, dy = c[:, 0] - ex, c[:, 1] - ey
    lon = dx * cs_ + dy * sn_; lat = -dx * sn_ + dy * cs_
    corr = (lon > 0.5) & (lon < length) & (np.abs(lat) < hw)
    lv = c[:, 11].astype(np.uint8)

    def rec(m, x, y, size_lv, zg, zt, zc, fl):
        r = np.zeros(len(x), REC)
        r["x"] = np.round((x - ox) / 0.025); r["y"] = np.round((y - oy) / 0.025)
        cm = lambda v: np.clip(np.round((v - z0) * 100), -32000, 32000)
        r["zg"] = cm(zg); r["zt"] = cm(zt); r["zc"] = cm(zc)
        r["levels"] = lv[m] | (size_lv.astype(np.uint8) << 4)
        r["cls"] = _CLS[c[m, 6].astype(np.int64)]
        r["flags"] = fl; r["conf"] = np.clip(c[m, 7] * 255, 0, 255)
        r["risk"] = np.clip(c[m, 12] * 255, 0, 255); r["neg"] = np.clip(c[m, 10] * 255, 0, 255)
        return r

    m = np.flatnonzero(occ | hole)
    g = c[m, 3]
    raised = rec(m, c[m, 0], c[m, 1], lv[m], g, np.where(occ[m], g + c[m, 4], g), g,
                 (1 + (dyn[m] & occ[m]) * 2 + hole[m] * 4 + corr[m] * 16).astype(np.uint8))
    m = np.flatnonzero(over)
    g = c[m, 3]
    overhead = rec(m, c[m, 0], c[m, 1], lv[m], g, g, g + c[m, 5], (8 + corr[m] * 16).astype(np.uint8))
    overhead["cls"] = 255
    # free ground: cells finer than `ground_lod` are drawn at that size (one representative per block)
    m = np.flatnonzero(~(occ | hole) & ((flags & E.C_GROUND) > 0))
    fine = c[m, 2] < ground_lod - 1e-6
    mf = m[fine]
    kx = np.floor(c[mf, 0] / ground_lod).astype(np.int64); ky = np.floor(c[mf, 1] / ground_lod).astype(np.int64)
    first = _one_per_key(kx, ky)
    mf = mf[first]; kx = kx[first]; ky = ky[first]
    lod_lv = int(round(np.log2(ground_lod / 0.05)))
    flat_a = rec(mf, (kx + 0.5) * ground_lod, (ky + 0.5) * ground_lod, np.full(len(mf), lod_lv), c[mf, 3], c[mf, 3], c[mf, 3],
                 (corr[mf] * 16).astype(np.uint8))
    mc = m[~fine]
    flat_b = rec(mc, c[mc, 0], c[mc, 1], lv[mc], c[mc, 3], c[mc, 3], c[mc, 3], (corr[mc] * 16).astype(np.uint8))
    blocks = [raised, np.concatenate([flat_a, flat_b]), overhead]

    # tiles for the minimap
    tx = np.floor(c[:, 0] / TILE).astype(np.int64) - int(np.round(ex / TILE))
    ty = np.floor(c[:, 1] / TILE).astype(np.int64) - int(np.round(ey / TILE))
    ok = (np.abs(tx) < 127) & (np.abs(ty) < 127)
    first = _one_per_key(tx[ok], ty[ok])
    tiles = np.zeros(len(first), np.dtype([("dx", "i1"), ("dy", "i1"), ("level", "u1"), ("risk", "u1")]))
    tiles["dx"] = tx[ok][first]; tiles["dy"] = ty[ok][first]; tiles["level"] = lv[ok][first]
    tiles["risk"] = np.clip(c[ok][first, 12] * 255, 0, 255)
    tpl = np.bincount(tiles["level"], minlength=5)[:5]

    tr = eng.tracks()      # id x y vx vy extent class dynamic hits path
    tracks = [{"id": int(r[0]), "cls": int(_CLS[int(r[6])]), "x": round(float(r[1] - ox), 2), "y": round(float(r[2] - oy), 2),
               "vx": round(float(r[3]), 2), "vy": round(float(r[4]), 2), "speed": round(float(np.hypot(r[3], r[4])), 2),
               "moving": bool(r[7] > 0), "ex": round(float(r[5]), 2), "ey": round(float(r[5]), 2),
               "dist": round(float(np.hypot(r[1] - ex, r[2] - ey)), 1)} for r in tr]
    head = {
        "frame": int(s["frame"]), "t": round(float(t), 2), "origin": [ox, oy, z0], "ego": [round(ex - ox, 3), round(ey - oy, 3), round(float(yaw), 4)],
        "speed": round(float(d.get("speed", 0.0)), 2), "n": [len(b) for b in blocks],
        "points": int(s["points"]), "net_points": int(s["ai_points"]),
        "memory_mb": round(s["map_bytes"] / 1e6, 3), "cells": int(s["cells_alloc"]),
        "cells_per_level": [int(s[f"level{i}"]) for i in range(5)], "tiles_per_level": [int(v) for v in tpl],
        "ms": {"triage": round(s["ms_ground"] + s["ms_triage"], 1), "network": round(s["ms_net"], 1), "policy": round(s["ms_risk"], 1),
               "fusion": round(s["ms_insert"], 1), "tracking": round(s["ms_track"], 1), "clearing": round(s["ms_rayclear"], 1),
               "total": round(s["ms_total"], 1)},
        "cleared": {"cleared": int(s["cleared"]), "stale": 0, "trail": 0}, "tracks": tracks, "policy": policy,
        "clearing_on": bool(ray_clearing), "corridor": [round(length, 1), hw], "tile": TILE,
        "alerts": {"hole": None, "block": None, "low": None},
        "decision": {"state": d.get("state", ""), "reason": d.get("reason", "")},
        "bytes_per_cell": 16, "engine": "C++ core", "seq": seq_no,
    }
    if extra:
        head.update(extra)
    hj = json.dumps(head, separators=(",", ":")).encode()
    hj += b" " * ((4 - len(hj) % 4) % 4)
    return b"".join([struct.pack("<I", len(hj)), hj, struct.pack("<I", len(tiles)), tiles.tobytes()] + [b.tobytes() for b in blocks])

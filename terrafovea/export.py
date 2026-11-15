"""Pack the live map into a compact binary frame for the dashboard.

Frame layout (little endian):
    uint32  json length, then UTF-8 JSON header (stats, tracks, alerts, block counts)
    uint32  tile count,  then 4 bytes per tile: int8 dx, int8 dy (tile units from ego), uint8 level, uint8 risk
    cells, 16 bytes each, in three consecutive blocks [raised | flat | overhead]:
        int16 x, y      cell centre, 2.5 cm units from the frame origin
        int16 zg, zt, zc  ground / top / ceiling height, cm above the road under the ego
        uint8 levels    low nibble = map level (resolution), high nibble = drawn size level
        uint8 cls       0-7, 255 = unknown
        uint8 flags     1 blocked, 2 dynamic, 4 hole, 8 overhead, 16 in driving corridor
        uint8 conf      class confidence 0-255
        uint8 risk      tile risk 0-255
        uint8 neg       negative-obstacle confidence 0-255

The 16-byte records are uploaded to the GPU unchanged as instance data, so the browser does no
per-cell work.  Free flat ground can be drawn coarser than it is stored (`ground_lod`) to keep
frames small; anything that blocks, moves, or is a suspected pit is always sent at map resolution.
"""
from __future__ import annotations

import json
import struct
import numpy as np

from .config import Config
from . import classes as C
from .grid import K, BYTES_PER_CELL

REC = np.dtype([("x", "<i2"), ("y", "<i2"), ("zg", "<i2"), ("zt", "<i2"), ("zc", "<i2"),
                ("levels", "u1"), ("cls", "u1"), ("flags", "u1"), ("conf", "u1"), ("risk", "u1"), ("neg", "u1")])
assert REC.itemsize == 16


def _cm(v, z0):
    return np.clip(np.round((np.nan_to_num(v) - z0) * 100), -32000, 32000).astype(np.int16)


def build_frame(pipe, out, view_radius=100.0, ground_lod=((20.0, 2), (1e9, 3)), extra=None) -> bytes:
    cfg: Config = pipe.cfg
    grid = pipe.grid
    pose = out["pose"]
    ts = cfg.tile_size
    ox = float(np.round(pose[0] / ts) * ts); oy = float(np.round(pose[1] / ts) * ts)
    z0 = float(pose[2] - cfg.sensor_height)
    c_, s_ = np.cos(pose[3]), np.sin(pose[3])
    risk_map = out["risk_map"]
    length = max(cfg.corridor_min_len, out["speed"] * cfg.corridor_horizon_s)
    raised, flat, over = [], [], []
    alerts = {"hole": None, "block": None, "low": None}

    def corridor(x, y):
        dx, dy = x - pose[0], y - pose[1]
        lon = dx * c_ + dy * s_; lat = -dx * s_ + dy * c_
        return (lon > 0.5) & (lon < length) & (np.abs(lat) < cfg.corridor_half_width), lon

    def emit(dst, x, y, zg, zt, zc, lv, dlv, cls, flags, conf, risk, neg):
        r = np.zeros(len(x), REC)
        r["x"] = np.round((x - ox) / 0.025); r["y"] = np.round((y - oy) / 0.025)
        r["zg"] = _cm(zg, z0); r["zt"] = _cm(zt, z0); r["zc"] = _cm(zc, z0)
        r["levels"] = lv | (dlv << 4); r["cls"] = np.where(cls >= 0, cls, 255)
        r["flags"] = flags; r["conf"] = np.clip(conf * 255, 0, 255); r["risk"] = risk; r["neg"] = neg
        dst.append(r)

    for L in grid.levels:
        t, slots, P = grid.level_view(L)
        if len(t) == 0:
            continue
        tcx = (grid.tx[t] + 0.5) * ts; tcy = (grid.ty[t] + 0.5) * ts
        td = np.hypot(tcx - pose[0], tcy - pose[1])
        keep = td <= view_radius
        t, slots, td = t[keep], slots[keep], td[keep]
        if len(t) == 0:
            continue
        trisk = (np.clip(risk_map[grid.tx[t] - grid.ox, grid.ty[t] - grid.oy], 0, 1) * 255).astype(np.uint8)
        zg = P["zg"][slots]; zt = P["zt"][slots]; zc = P["zc"][slots]; lo = P["lo"][slots]
        ng = P["ng"][slots]; neg = P["neg"][slots]; dyn = P["dyn"][slots]; hist = P["hist"][slots].astype(np.int32)
        has_g = np.isfinite(zg)
        occ = (lo > 0) & np.isfinite(zt)
        zref = np.where(has_g, zg, z0)
        hole = neg >= int(cfg.neg_report * 255)
        blocked = (occ & (((zt - zref) >= cfg.step_height) | ~has_g)) | hole
        hz = occ | hole
        ho, hg = hist[:, :, 2:], hist[:, :, :2]
        to, tg = ho.sum(2), hg.sum(2)
        cls_o = 2 + ho.argmax(2); conf_o = ho.max(2) / np.maximum(to, 1)
        cls_g = np.where(tg > 0, hg.argmax(2), -1); conf_g = hg.max(2) / np.maximum(tg, 1)

        # ---- raised cells and pits: always at map resolution ----
        ti, ci = np.nonzero(hz)
        if len(ti):
            x, y = grid.cell_xy(L, t[ti], ci)
            inc, lon = corridor(x, y)
            o_ = occ[ti, ci]; h_ = hole[ti, ci]
            fl = (blocked[ti, ci] * 1 + ((dyn[ti, ci] > 0) & o_) * 2 + h_ * 4 + inc * 16).astype(np.uint8)
            cls = np.where(o_ & (to[ti, ci] > 0), cls_o[ti, ci], cls_g[ti, ci])
            cf = np.where(o_, conf_o[ti, ci], conf_g[ti, ci])
            zr = zref[ti, ci]
            emit(raised, x, y, zr, np.where(o_, zt[ti, ci], zr), zr, L, L, cls, fl, cf, trisk[ti], neg[ti, ci])
            for key, m in (("hole", h_ & inc), ("block", o_ & blocked[ti, ci] & inc & ~h_)):
                if m.any():
                    d = float(lon[m].min())
                    if alerts[key] is None or d < alerts[key]:
                        alerts[key] = d
        # ---- overhead surface above free space ----
        ti, ci = np.nonzero(np.isfinite(zc) & ~occ)
        if len(ti):
            x, y = grid.cell_xy(L, t[ti], ci)
            inc, lon = corridor(x, y)
            zr = zref[ti, ci]
            emit(over, x, y, zr, zr, zc[ti, ci], L, L, np.full(len(ti), -1), (8 + inc * 16).astype(np.uint8),
                 np.ones(len(ti)), trisk[ti], np.zeros(len(ti), np.uint8))
            low = inc & ((zc[ti, ci] - zr) < cfg.vehicle_height + 1.0)
            if low.any():
                d = float(lon[low].min())
                alerts["low"] = d if alerts["low"] is None else min(alerts["low"], d)
        # ---- free ground, optionally drawn coarser ----
        free = (ng > 0) & ~hz
        want = np.full(len(t), L, np.int64)
        prev = 0.0
        for dmax, lv in ground_lod or ():
            want = np.where((td >= prev) & (td < dmax), np.maximum(L, lv), want); prev = dmax
        want = np.minimum(want, grid.shift)
        for D in np.unique(want):
            sel = np.flatnonzero(want == D)
            f = free[sel]
            zz = np.where(f, zg[sel], 0.0); w = f.astype(np.float32)
            cg = cls_g[sel]; cf = conf_g[sel]
            r0 = (cg == C.ROAD) & f; r1 = (cg == C.TERRAIN) & f
            k = 4 ** int(D - L)
            if k > 1:
                q = lambda a: a.reshape(len(sel), -1, k)
                wsum = q(w).sum(2)
                zz = q(zz).sum(2) / np.maximum(wsum, 1); f = wsum > 0
                n0, n1 = q(r0.astype(np.float32)).sum(2), q(r1.astype(np.float32)).sum(2)
                cg = np.where(n0 + n1 == 0, -1, np.where(n1 > n0, C.TERRAIN, C.ROAD))
                cf = (q(cf * w).sum(2) / np.maximum(wsum, 1))
            ti, ci = np.nonzero(f)
            if len(ti) == 0:
                continue
            x, y = grid.cell_xy(int(D), t[sel][ti], ci)
            inc, _ = corridor(x, y)
            z = zz[ti, ci]
            emit(flat, x, y, z, z, z, L, int(D), cg[ti, ci], (inc * 16).astype(np.uint8), cf[ti, ci], trisk[sel][ti],
                 np.zeros(len(ti), np.uint8))

    blocks = [np.concatenate(b) if b else np.zeros(0, REC) for b in (raised, flat, over)]
    ids = grid.tile_ids()
    tdx = grid.tx[ids] - int(np.round(pose[0] / ts)); tdy = grid.ty[ids] - int(np.round(pose[1] / ts))
    okt = (np.abs(tdx) < 127) & (np.abs(tdy) < 127)
    tiles = np.zeros(int(okt.sum()), np.dtype([("dx", "i1"), ("dy", "i1"), ("level", "u1"), ("risk", "u1")]))
    tiles["dx"] = tdx[okt]; tiles["dy"] = tdy[okt]; tiles["level"] = grid.tlevel[ids][okt]
    tiles["risk"] = (np.clip(risk_map[grid.tx[ids][okt] - grid.ox, grid.ty[ids][okt] - grid.oy], 0, 1) * 255).astype(np.uint8)

    tm = out["timing"]
    tracks = [{"id": t["id"], "cls": int(t["cls"]), "x": round(t["x"] - ox, 2), "y": round(t["y"] - oy, 2),
               "vx": round(t["vx"], 2), "vy": round(t["vy"], 2), "speed": round(t["speed"], 2),
               "moving": bool(t["moving"]), "ex": round(t["ex"], 2), "ey": round(t["ey"], 2),
               "dist": round(float(np.hypot(t["x"] - pose[0], t["y"] - pose[1])), 1)}
              for t in out["tracks"] if t["confirmed"] and t["visible"]]
    head = {
        "frame": out["frame"], "t": round(out["t"], 2), "origin": [ox, oy, z0],
        "ego": [round(pose[0] - ox, 3), round(pose[1] - oy, 3), round(float(pose[3]), 4)], "speed": round(out["speed"], 2),
        "n": [len(b) for b in blocks], "points": out["n_points"], "net_points": out["n_net"],
        "memory_mb": round(out["memory_bytes"] / 1e6, 3), "cells": out["n_cells"],
        "cells_per_level": [int(v) for v in out["cells_per_level"]], "tiles_per_level": [int(v) for v in out["tiles_per_level"]],
        "ms": {k: round(v * 1000, 1) for k, v in tm.items()},
        "cleared": out["clearing"], "tracks": tracks, "policy": cfg.policy, "clearing_on": bool(cfg.ray_clearing),
        "corridor": [round(length, 1), cfg.corridor_half_width], "bands": list(cfg.bands), "tile": ts,
        "alerts": {k: (None if v is None else round(v, 1)) for k, v in alerts.items()},
        "bytes_per_cell": BYTES_PER_CELL,
    }
    if extra:
        head.update(extra)
    hj = json.dumps(head, separators=(",", ":")).encode()
    hj += b" " * ((4 - len(hj) % 4) % 4)
    return b"".join([struct.pack("<I", len(hj)), hj, struct.pack("<I", len(tiles)), tiles.tobytes()] +
                    [b.tobytes() for b in blocks])

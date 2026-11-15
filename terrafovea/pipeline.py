"""The full per-sweep pipeline.

    raw sweep -> geometric triage -> selective neural segmentation -> risk plan + exact regrid
              -> multi-surface fusion -> negative-obstacle evidence -> tracking -> ghost clearing
"""
from __future__ import annotations

import time
import numpy as np

from .config import Config
from . import classes as C
from . import ground
from .grid import AdaptiveGrid, KIND_GROUND, KIND_OBST, KIND_OVER, KIND_BELOW, BYTES_PER_CELL
from .policy import ResolutionPolicy
from .visibility import RangeImage, clear_ghosts, clear_trails
from .tracking import Tracker, cluster

STAGES = ("triage", "network", "policy", "fusion", "tracking", "clearing")


class Pipeline:
    def __init__(self, cfg: Config | None = None, net=None, fov=(None, None)):
        self.cfg = cfg or Config()
        self.net = net if self.cfg.use_network else None
        self.grid = AdaptiveGrid(self.cfg)
        self.policy = ResolutionPolicy(self.cfg)
        self.tracker = Tracker(self.cfg)
        self.tracks = []
        self.frame = 0
        self.fov = fov
        self._last_pose = None
        self._last_t = None
        self.speed = 0.0

    def process(self, sweep: dict, oracle: bool = False) -> dict:
        """sweep: {'points': (N,3) sensor frame, 'intensity': (N,), 'pose': [x, y, z, yaw], 't': seconds}"""
        cfg = self.cfg
        tm = {}
        t0 = time.perf_counter()
        pose = np.asarray(sweep["pose"], dtype=np.float64)
        pts = sweep["points"]
        if "T" in sweep:                       # full 6-DoF pose (real data): respect roll and pitch
            T = np.asarray(sweep["T"], dtype=np.float64)
            pw = pts.astype(np.float64) @ T[:3, :3].T + T[:3, 3]
            xw, yw, zw = pw[:, 0], pw[:, 1], pw[:, 2].astype(np.float32)
        else:
            c, s = np.cos(pose[3]), np.sin(pose[3])
            xw = pose[0] + c * pts[:, 0].astype(np.float64) - s * pts[:, 1]
            yw = pose[1] + s * pts[:, 0].astype(np.float64) + c * pts[:, 1]
            zw = (pose[2] + pts[:, 2]).astype(np.float32)
        t_now = float(sweep.get("t", self.frame * 0.1))
        dt = 0.1
        if self._last_pose is not None:
            dt = max(t_now - self._last_t, 1e-3)
            self.speed = float(np.hypot(pose[0] - self._last_pose[0], pose[1] - self._last_pose[1]) / dt)
        self._last_pose, self._last_t = pose, t_now

        # ---- 1. geometric triage -------------------------------------------------
        g = ground.segment(xw, yw, zw, pose[:2], cfg, sensor_z=pose[2])
        kind = g["kind"]; h = g["h"]; rng = g["range"]
        inr = rng <= cfg.max_range
        cls = np.where(kind == KIND_GROUND, g["gcls"], -1).astype(np.int8)
        conf = np.where(cls >= 0, 0.9, 0.0).astype(np.float32)
        tm["triage"] = time.perf_counter() - t0; t1 = time.perf_counter()

        # ---- 2. neural segmentation, only where geometry cannot decide -------------
        n_net = 0
        if oracle:
            cls = sweep["label"].astype(np.int8).copy()
            conf[:] = 1.0
        elif self.net is not None:
            if cfg.selective_ai:
                m = (kind != KIND_GROUND) | g["uncertain"]
            else:
                m = np.ones(len(xw), bool)
            m &= inr
            pc, pf = self.net.predict(xw[m], yw[m], zw[m], h[m], sweep["intensity"][m], rng[m])
            n_net = getattr(self.net, "last_n", int(m.sum()))
            pc = pc.copy()
            km = kind[m]
            isg = km == KIND_GROUND
            pc[isg & (pc > C.TERRAIN)] = -1                 # a ground return can only be road or terrain
            keep_geo = isg & (g["gcls"][m] >= 0)            # geometry was already sure
            pc[keep_geo] = g["gcls"][m][keep_geo]
            pc[(km == KIND_OBST) & (pc == C.ROAD)] = C.OTHER
            cls[m] = pc; conf[m] = np.where(pc >= 0, pf, 0.0)
        else:
            cls[kind == KIND_OBST] = C.OTHER
        tm["network"] = time.perf_counter() - t1; t1 = time.perf_counter()

        # ---- 3. resolution plan + exact regrid ------------------------------------
        self.grid.set_window(pose[0], pose[1])
        level_map, risk_map = self.policy.plan(self.grid, pose, self.speed, self.tracks)
        self.policy.apply(self.grid, level_map, risk_map, pose)
        tm["policy"] = time.perf_counter() - t1; t1 = time.perf_counter()

        # ---- 4. multi-surface fusion + negative-obstacle evidence -----------------
        self.grid.insert(xw[inr], yw[inr], zw[inr], kind[inr], cls[inr], self.frame, level_map)
        b = (kind == KIND_BELOW) & (rng < 30.0) & (h > -2.0)
        # only where the ground plane is well supported: next to walls and parked cars the plane is a guess
        b &= g["support"].reshape(-1)[g["lin"]] >= cfg.neg_min_support       # far or absurdly deep 'pits' are not trusted
        n_below = int(b.sum())
        if n_below:
            depth = -h[b] - (cfg.neg_tau0 + cfg.neg_tau_r * rng[b])
            amt = (np.clip(depth / 0.10, 0.15, 1.0) * cfg.neg_gain).astype(np.float32)
            hs = np.maximum(pose[2] - (zw[b] - h[b]), 0.5)        # sensor height above the local plane
            frac = hs / (hs - h[b])                               # where the beam crossed the surface
            for f_ in (1.0, 0.5):
                ff = frac + (1 - frac) * (1 - f_) if f_ < 1 else frac
                self.grid.add_negative(pose[0] + ff * (xw[b] - pose[0]), pose[1] + ff * (yw[b] - pose[1]), amt)
            self.grid.add_negative(xw[b], yw[b], amt)
        tm["fusion"] = time.perf_counter() - t1; t1 = time.perf_counter()

        # ---- 5. dynamic objects ---------------------------------------------------
        if self.net is not None or oracle:
            cand = (kind == KIND_OBST) & ((cls == C.VEHICLE) | (cls == C.PEDESTRIAN)) & (rng < 90.0)
        else:
            cand = (kind == KIND_OBST) & (h > 0.3) & (rng < 90.0)
        ci = np.flatnonzero(cand)
        pid, dets = cluster(xw[ci], yw[ci], cls[ci], pose[:2], cfg)
        self.tracks = self.tracker.step(dets, dt)
        moving_c = {t["cid"] for t in self.tracks if t["moving"] and t["cid"] >= 0}
        point_dyn = np.zeros(len(xw), bool)
        if moving_c:
            mm = np.isin(pid, list(moving_c))
            point_dyn[ci[mm]] = True
            self.grid.mark_dynamic(xw[ci[mm]], yw[ci[mm]], self.frame)
        tm["tracking"] = time.perf_counter() - t1; t1 = time.perf_counter()

        # ---- 6. ghost clearing ------------------------------------------------------
        clr = {"seen_through": 0, "cleared": 0, "stale": 0, "trail": 0}
        if cfg.ray_clearing:
            ri = RangeImage(pts, self.fov[0], self.fov[1])
            clr = clear_ghosts(self.grid, ri, pose, self.frame, cfg)
            clr["trail"] = clear_trails(self.grid, self.tracks, self.frame, cfg)
        tm["clearing"] = time.perf_counter() - t1
        tm["total"] = time.perf_counter() - t0

        tiles, cells = self.grid.level_counts()
        out = {
            "frame": self.frame, "t": t_now, "pose": pose, "speed": self.speed, "timing": tm,
            "n_points": int(len(xw)), "n_net": int(n_net), "n_below": n_below,
            "kind": kind, "cls": cls, "conf": conf, "point_dyn": point_dyn, "range": rng, "h": h,
            "tracks": self.tracks, "clearing": clr,
            "memory_bytes": self.grid.memory_bytes(), "n_cells": int(cells.sum()),
            "tiles_per_level": tiles, "cells_per_level": cells,
            "level_map": level_map, "risk_map": risk_map,
        }
        self.frame += 1
        return out


def cell_state(cells: dict, cfg: Config, ground_z: float = 0.0):
    """Derive what a planner needs from the raw cell fields."""
    zg = cells["zg"]; zt = cells["zt"]; zc = cells["zc"]
    has_g = np.isfinite(zg)
    occ = (cells["lo"] > 0) & np.isfinite(zt)
    zref = np.where(has_g, zg, np.where(occ, np.minimum(zt, ground_z), ground_z))
    step = np.where(occ, zt - zref, 0.0)
    neg_conf = cells["neg"].astype(np.float32) / 255.0
    hole = neg_conf >= cfg.neg_report
    blocked = (occ & ((step >= cfg.step_height) | ~has_g)) | hole
    overhead = np.isfinite(zc)
    clearance = np.where(overhead, zc - zref, np.inf)
    return {"blocked": blocked, "occupied": occ, "hole": hole, "neg_conf": neg_conf, "overhead": overhead & ~occ,
            "clearance": clearance, "height": step, "zref": zref,
            "dynamic": (cells["dyn"] > 0) & occ}

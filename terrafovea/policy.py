"""Where should the map spend its cells?

Three ideas, in order:

1. Sensor physics sets the distance bands.  A 0.2 deg LiDAR puts neighbouring
   returns r * 0.0035 m apart, so beyond ~14 m a 5 cm cell cannot be filled by
   one sweep.  Fine cells far away store nothing new - that is why resolution
   may fall with distance without losing information.

2. Risk overrides distance.  A tile is refined when it lies in the path the
   vehicle is about to drive, holds an obstacle or a suspected pit in that
   path, holds a moving object, or lies where a moving object is predicted to
   be within the next few seconds.

3. Memory is a hard budget.  If the scene asks for more cells than the budget,
   the lowest-risk, farthest tiles are coarsened first.  Memory stays bounded
   in any scene.

Refining is immediate; coarsening waits a few frames (hysteresis), so tiles do
not flicker between levels and safety-relevant detail is never dropped early.
"""
import numpy as np

from .config import Config
from . import classes as C


class ResolutionPolicy:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last_risk = None
        self.last_levels = None

    def base_levels(self, dist):
        cfg = self.cfg
        if cfg.policy == "uniform_fine":
            return np.zeros(dist.shape, np.int8)
        if cfg.policy == "uniform_coarse":
            return np.full(dist.shape, cfg.far_level, np.int8)
        lv = np.searchsorted(np.asarray(cfg.bands), dist, side="right")
        return np.minimum(lv, cfg.far_level).astype(np.int8)

    def physics_floor(self, dist):
        cfg = self.cfg
        spacing = np.maximum(dist, 1e-3) * np.deg2rad(cfg.az_res_deg)
        lv = np.ceil(np.log2(np.maximum(spacing / cfg.base_res, 1.0)))
        return np.clip(lv - cfg.physics_floor_slack, 0, cfg.max_level).astype(np.int8)

    def plan(self, grid, pose, speed, tracks):
        """Return (level_map, risk_map) over the grid's lookup window."""
        cfg = self.cfg
        ts = cfg.tile_size
        Wn = grid.Wn
        cx = (grid.ox + np.arange(Wn) + 0.5) * ts
        cy = (grid.oy + np.arange(Wn) + 0.5) * ts
        X, Y = np.meshgrid(cx, cy, indexing="ij")
        dx, dy = X - pose[0], Y - pose[1]
        dist = np.hypot(dx, dy)
        base = self.base_levels(dist)
        risk = np.zeros((Wn, Wn), np.float32)
        if cfg.policy != "risk":
            self.last_risk, self.last_levels = risk, base
            return base, risk

        c, s = np.cos(pose[3]), np.sin(pose[3])
        lon = dx * c + dy * s; lat = -dx * s + dy * c
        length = max(cfg.corridor_min_len, speed * cfg.corridor_horizon_s)
        in_corr = (lon > -2.0) & (lon < length) & (np.abs(lat) < cfg.corridor_half_width + ts * 0.5)
        risk[in_corr] = 0.3                                    # the path itself: one level finer

        # what the map already knows about each tile
        occ, neg, dyn = grid.tile_summary()
        ids = grid.tile_ids()
        wx, wy = grid.tx[ids] - grid.ox, grid.ty[ids] - grid.oy
        ok = (wx >= 0) & (wx < Wn) & (wy >= 0) & (wy < Wn)
        ids, wx, wy = ids[ok], wx[ok], wy[ok]
        corr_t = in_corr[wx, wy]
        r = risk[wx, wy]
        r = np.where(corr_t & occ[ids], np.maximum(r, 0.6), r)          # obstacle in the path
        r = np.where(corr_t & (neg[ids] >= 30), np.maximum(r, 0.8), r)  # suspected pit in the path
        r = np.where(~corr_t & (neg[ids] >= 90), np.maximum(r, 0.3), r)
        r = np.where(dyn[ids], np.maximum(r, 0.55), r)                  # something is moving here
        risk[wx, wy] = r

        # predicted motion of tracked objects
        for tr in tracks:
            if not tr["confirmed"]:
                continue
            rad = 1.2 if tr["cls"] == C.PEDESTRIAN else 2.6
            if tr["moving"]:
                taus = np.arange(0.0, cfg.predict_horizon_s + 1e-6, 0.25)
            elif tr["cls"] == C.PEDESTRIAN:
                taus = np.array([0.0])                                   # a standing person may step out
            else:
                continue
            px = tr["x"] + tr["vx"] * taus; py = tr["y"] + tr["vy"] * taus
            for k in range(len(taus)):
                ix = int(np.floor(px[k] / ts)) - grid.ox; iy = int(np.floor(py[k] / ts)) - grid.oy
                rr = int(np.ceil(rad / ts))
                x0, x1 = max(ix - rr, 0), min(ix + rr + 1, Wn)
                y0, y1 = max(iy - rr, 0), min(iy + rr + 1, Wn)
                if x0 >= x1 or y0 >= y1:
                    continue
                sub = (X[x0:x1, y0:y1] - px[k]) ** 2 + (Y[x0:x1, y0:y1] - py[k]) ** 2 <= (rad + ts * 0.7) ** 2
                hits_path = in_corr[x0:x1, y0:y1] & sub
                val = np.where(hits_path, 0.9, 0.55 if taus[k] <= 1.0 else 0.3)
                risk[x0:x1, y0:y1] = np.where(sub, np.maximum(risk[x0:x1, y0:y1], val), risk[x0:x1, y0:y1])

        boost = np.searchsorted(np.asarray(cfg.risk_thresholds), risk, side="right").astype(np.int8)
        floor = self.physics_floor(dist)
        lv = np.minimum(base, np.maximum(base - boost, floor)).astype(np.int8)
        self.last_risk, self.last_levels = risk, lv
        return lv, risk

    def apply(self, grid, level_map, risk_map, pose):
        """Regrid live tiles toward the plan, with hysteresis and the memory budget."""
        cfg = self.cfg
        ids = grid.tile_ids()
        if len(ids) == 0:
            return
        wx, wy = grid.tx[ids] - grid.ox, grid.ty[ids] - grid.oy
        target = level_map[wx, wy].astype(np.int8)
        cur = grid.tlevel[ids]
        finer = target < cur
        coarser = target > cur
        grid.wait[ids[~coarser]] = 0
        grid.wait[ids[coarser]] = np.minimum(grid.wait[ids[coarser]] + 1, 250)
        go = finer | (coarser & (grid.wait[ids] >= cfg.coarsen_patience))
        new = np.where(go, target, cur).astype(np.int8)

        # hard memory budget: coarsen lowest-risk, farthest tiles first
        T = grid.T
        cells = (T >> new.astype(np.int64)) ** 2
        over = int(cells.sum()) - cfg.cell_budget
        passes = 0
        while cfg.policy == "risk" and over > 0 and passes < cfg.max_level:
            ts = cfg.tile_size
            d = np.hypot((grid.tx[ids] + 0.5) * ts - pose[0], (grid.ty[ids] + 0.5) * ts - pose[1])
            prio = risk_map[wx, wy] * 1000.0 - d
            cand = np.flatnonzero(new < cfg.max_level)
            order = cand[np.argsort(prio[cand], kind="stable")]
            saving = cells[order] * 3 // 4
            k = int(np.searchsorted(np.cumsum(saving), over, side="left")) + 1
            pick = order[:k]
            new[pick] += 1
            cells = (T >> new.astype(np.int64)) ** 2
            over = int(cells.sum()) - cfg.cell_budget
            passes += 1
        ch = new != cur
        grid.regrid(ids[ch], new[ch])
        grid.wait[ids[ch]] = 0

"""Variable-resolution 2.5D grid engine.

The world is cut into square tiles (1.6 m).  Each tile owns a dense block of
cells at ONE pyramid level (5, 10, 20, 40 or 80 cm), stored in Z-order.  Tiles
that have never been observed do not exist, so empty space costs nothing.

Every cell is a small multi-surface column:

    zg    ground elevation (running mean of ground returns)            float32
    ng    ground evidence count                                        uint16
    zt    top of the obstacle that blocks the vehicle (below 2.5 m)    float32
    zc    underside of anything overhead (ceiling / clearance)         float32
    lo    occupancy log-odds                                           int8
    hist  per-class evidence (8 classes)                               8 x uint8
    dyn   dynamic evidence                                             uint8
    neg   negative-obstacle confidence                                 uint8
    stamp last frame an obstacle was seen here                         uint16
                                                             total     27 bytes

Changing a tile's resolution is exact:

    coarsen  -> zt = max of children, zc = min, lo = max, hist = per-class max,
                zg = evidence-weighted mean.  Nothing that blocks the vehicle
                can disappear, and a rare class is never out-voted.
    refine   -> children inherit the parent; inherited occupancy is marked
                tentative so one contradicting ray can remove it.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from . import classes as C
from . import morton

K = C.NUM_CLASSES
BYTES_PER_CELL = 4 + 2 + 4 + 4 + 1 + K + 1 + 1 + 2
TILE_OVERHEAD = 16
NG_CAP = 60          # running-mean memory for ground height
KIND_GROUND, KIND_OBST, KIND_OVER, KIND_BELOW = 0, 1, 2, 3

_FIELDS = (("zg", np.float32, np.nan), ("ng", np.uint16, 0), ("zt", np.float32, np.nan),
           ("zc", np.float32, np.nan), ("lo", np.int8, 0), ("dyn", np.uint8, 0),
           ("neg", np.uint8, 0), ("stamp", np.uint16, 0))


def _group(idx):
    """Sort-based grouping: returns (order, unique_values, start_offsets)."""
    order = np.argsort(idx, kind="stable")
    s = idx[order]
    starts = np.flatnonzero(np.concatenate(([True], s[1:] != s[:-1])))
    return order, s[starts], starts


class _Pool:
    """All tiles of one level, stored as (capacity, cells_per_tile) arrays."""

    def __init__(self, n_side: int, cap: int = 64):
        self.n = n_side
        self.ncell = n_side * n_side
        self.cap = 0
        self.free: list[int] = []
        self.a: dict[str, np.ndarray] = {}
        self._grow(cap)

    def _grow(self, new_cap: int):
        for name, dt, init in _FIELDS:
            blk = np.full((new_cap - self.cap, self.ncell), init, dtype=dt)
            self.a[name] = blk if self.cap == 0 else np.concatenate([self.a[name], blk])
        blk = np.zeros((new_cap - self.cap, self.ncell, K), dtype=np.uint8)
        self.a["hist"] = blk if self.cap == 0 else np.concatenate([self.a["hist"], blk])
        self.free.extend(range(new_cap - 1, self.cap - 1, -1))
        self.cap = new_cap

    def alloc(self, m: int) -> np.ndarray:
        while len(self.free) < m:
            self._grow(max(self.cap * 2, self.cap + m))
        out = np.array([self.free.pop() for _ in range(m)], dtype=np.int64)
        return out

    def release(self, slots: np.ndarray):
        if len(slots) == 0:
            return
        for name, _, init in _FIELDS:
            self.a[name][slots] = init
        self.a["hist"][slots] = 0
        self.free.extend(int(s) for s in slots)


# ---------------------------------------------------------------------------
# exact resampling
# ---------------------------------------------------------------------------
def coarsen_block(f: dict) -> dict:
    """(m, ncell[, K]) -> (m, ncell/4[, K]).  Children are consecutive in Z-order."""
    m = f["zg"].shape[0]
    q = lambda a: a.reshape(m, -1, 4)
    ng = q(f["ng"]).astype(np.float64)        # float64: makes refine-then-coarsen bit-exact
    zg = q(f["zg"]).astype(np.float64)
    w = np.where(np.isfinite(zg), ng, 0.0)
    wsum = w.sum(2)
    with np.errstate(invalid="ignore", divide="ignore"):
        zg_new = np.where(wsum > 0, (np.nan_to_num(zg) * w).sum(2) / wsum, np.nan).astype(np.float32)
    out = {"zg": zg_new, "ng": q(f["ng"]).max(2)}
    zt = q(f["zt"]); zc = q(f["zc"])
    out["zt"] = np.where(np.isfinite(zt).any(2), np.nanmax(np.where(np.isfinite(zt), zt, -np.inf), axis=2), np.nan).astype(np.float32)
    out["zc"] = np.where(np.isfinite(zc).any(2), np.nanmin(np.where(np.isfinite(zc), zc, np.inf), axis=2), np.nan).astype(np.float32)
    for name in ("lo", "dyn", "neg", "stamp"):
        out[name] = q(f[name]).max(2)
    out["hist"] = f["hist"].reshape(m, -1, 4, K).max(2)
    return out


def refine_block(f: dict, lo_tentative: int) -> dict:
    """(m, ncell[, K]) -> (m, 4*ncell[, K]).  Children inherit the parent."""
    out = {name: np.repeat(f[name], 4, axis=1) for name in ("zg", "ng", "zt", "zc", "dyn", "neg", "stamp", "hist")}
    out["lo"] = np.repeat(np.minimum(f["lo"], lo_tentative).astype(np.int8), 4, axis=1)
    return out


class AdaptiveGrid:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.T = cfg.tile_cells
        self.shift = int(np.log2(self.T))
        assert (1 << self.shift) == self.T, "tile_cells must be a power of two"
        assert cfg.max_level <= self.shift
        self.levels = list(range(cfg.max_level + 1))
        self.pools = [_Pool(self.T >> L) for L in self.levels]
        self.lut, _, _ = morton.tables(self.T)
        self.inv = [morton.tables(self.T >> L)[1:] for L in self.levels]
        cap = 1024
        self.tx = np.zeros(cap, np.int32); self.ty = np.zeros(cap, np.int32)
        self.tlevel = np.zeros(cap, np.int8); self.tslot = np.zeros(cap, np.int64)
        self.alive = np.zeros(cap, bool); self.wait = np.zeros(cap, np.uint8)
        self.tfree = list(range(cap - 1, -1, -1))
        half = int(np.ceil(cfg.keep_radius / cfg.tile_size)) + 2
        self.Wn = 2 * half
        self.half = half
        self.W = np.full((self.Wn, self.Wn), -1, np.int32)
        self.ox = 0; self.oy = 0
        self.stats = {"split": 0, "merge": 0, "created": 0, "dropped": 0}

    # ------------------------------------------------------------------ tiles
    def _grow_tiles(self):
        cap = len(self.tx)
        for name in ("tx", "ty", "tlevel", "tslot", "alive", "wait"):
            a = getattr(self, name)
            setattr(self, name, np.concatenate([a, np.zeros(cap, a.dtype)]))
        self.tfree.extend(range(2 * cap - 1, cap - 1, -1))

    def set_window(self, ego_x: float, ego_y: float):
        """Re-centre the lookup window on the ego and drop tiles that fell out of range."""
        ts = self.cfg.tile_size
        self.ox = int(np.floor(ego_x / ts)) - self.half
        self.oy = int(np.floor(ego_y / ts)) - self.half
        ids = np.flatnonzero(self.alive)
        cx = (self.tx[ids] + 0.5) * ts - ego_x; cy = (self.ty[ids] + 0.5) * ts - ego_y
        far = cx * cx + cy * cy > self.cfg.keep_radius ** 2
        self._drop(ids[far])
        ids = ids[~far]
        self.W.fill(-1)
        self.W[self.tx[ids] - self.ox, self.ty[ids] - self.oy] = ids

    def _drop(self, ids):
        if len(ids) == 0:
            return
        for L in self.levels:
            m = self.tlevel[ids] == L
            self.pools[L].release(self.tslot[ids[m]])
        self.alive[ids] = False
        self.tfree.extend(int(i) for i in ids)
        self.stats["dropped"] += len(ids)

    def create(self, wx: np.ndarray, wy: np.ndarray, levels: np.ndarray):
        """Create tiles at window coordinates (wx, wy) with the given levels."""
        m = len(wx)
        if m == 0:
            return
        while len(self.tfree) < m:
            self._grow_tiles()
        ids = np.array([self.tfree.pop() for _ in range(m)], dtype=np.int64)
        self.tx[ids] = wx + self.ox; self.ty[ids] = wy + self.oy
        self.tlevel[ids] = levels; self.alive[ids] = True; self.wait[ids] = 0
        for L in self.levels:
            sel = levels == L
            if sel.any():
                self.tslot[ids[sel]] = self.pools[L].alloc(int(sel.sum()))
        self.W[wx, wy] = ids
        self.stats["created"] += m

    # ---------------------------------------------------------------- regrid
    def regrid(self, ids: np.ndarray, new_levels: np.ndarray):
        """Move tiles to new levels with exact coarsen/refine."""
        if len(ids) == 0:
            return
        cur = self.tlevel[ids]
        for a in np.unique(cur):
            for b in np.unique(new_levels[cur == a]):
                if a == b:
                    continue
                sel = ids[(cur == a) & (new_levels == b)]
                pa = self.pools[a]
                slots = self.tslot[sel]
                blk = {k: v[slots] for k, v in pa.a.items()}
                lv = int(a)
                while lv < b:
                    blk = coarsen_block(blk); lv += 1
                while lv > b:
                    blk = refine_block(blk, self.cfg.lo_hit); lv -= 1
                pa.release(slots)
                pb = self.pools[b]
                ns = pb.alloc(len(sel))
                for k, v in blk.items():
                    pb.a[k][ns] = v
                self.tslot[sel] = ns; self.tlevel[sel] = b
                self.stats["merge" if b > a else "split"] += len(sel)

    # ---------------------------------------------------------------- lookup
    def locate(self, x: np.ndarray, y: np.ndarray):
        """World xy -> (valid mask, tile id, level, flat cell index within the level pool)."""
        inv = 1.0 / self.cfg.base_res
        gx = np.floor(x * inv).astype(np.int64); gy = np.floor(y * inv).astype(np.int64)
        wx = (gx >> self.shift) - self.ox; wy = (gy >> self.shift) - self.oy
        ok = (wx >= 0) & (wx < self.Wn) & (wy >= 0) & (wy < self.Wn)
        tid = np.full(x.shape, -1, np.int64)
        tid[ok] = self.W[wx[ok], wy[ok]]
        ok &= tid >= 0
        lvl = np.zeros(x.shape, np.int64); flat = np.zeros(x.shape, np.int64)
        t = tid[ok]
        L = self.tlevel[t].astype(np.int64)
        mi = self.lut[gy[ok] & (self.T - 1), gx[ok] & (self.T - 1)].astype(np.int64) >> (2 * L)
        ncell = (self.T >> L) ** 2
        lvl[ok] = L; flat[ok] = self.tslot[t] * ncell + mi
        return ok, tid, lvl, flat

    def window_coords(self, x, y):
        inv = 1.0 / self.cfg.tile_size
        wx = np.floor(x * inv).astype(np.int64) - self.ox
        wy = np.floor(y * inv).astype(np.int64) - self.oy
        ok = (wx >= 0) & (wx < self.Wn) & (wy >= 0) & (wy < self.Wn)
        return wx, wy, ok

    # ---------------------------------------------------------------- insert
    def insert(self, x, y, z, kind, cls, frame: int, level_map: np.ndarray):
        """Fuse one classified sweep.  `level_map[wx, wy]` gives the level for new tiles."""
        cfg = self.cfg
        wx, wy, ok = self.window_coords(x, y)
        new = ok.copy()
        new[ok] = self.W[wx[ok], wy[ok]] < 0
        if new.any():
            key = np.unique(wx[new] * self.Wn + wy[new])
            nx, ny = key // self.Wn, key % self.Wn
            self.create(nx, ny, level_map[nx, ny].astype(np.int8))
        ok, tid, lvl, flat = self.locate(x, y)
        stamp = np.uint16(frame & 0xFFFF)
        for L in self.levels:
            sel = ok & (lvl == L)
            if not sel.any():
                continue
            P = self.pools[L].a
            f = flat[sel]; zz = z[sel]; kk = kind[sel]; cc = cls[sel]
            # ---- ground surface (includes returns from below the surface: the pit floor) ----
            g = (kk == KIND_GROUND) | (kk == KIND_BELOW)
            if g.any():
                order, cells, starts = _group(f[g])
                zs = zz[g][order]
                s = np.add.reduceat(zs.astype(np.float64), starts)
                cnt = np.diff(np.concatenate((starts, [len(zs)])))
                ng = P["ng"].reshape(-1); zg = P["zg"].reshape(-1)
                n0 = np.minimum(ng[cells], NG_CAP).astype(np.float64)
                z0 = np.where(n0 > 0, zg[cells], 0.0)
                zg[cells] = ((z0 * n0 + s) / (n0 + cnt)).astype(np.float32)
                ng[cells] = np.minimum(ng[cells].astype(np.int64) + cnt, 65535).astype(np.uint16)
                c0 = cells[np.minimum.reduceat(kk[g][order], starts) == KIND_GROUND]   # saw the real surface
                neg = P["neg"].reshape(-1)
                neg[c0] = np.maximum(neg[c0].astype(np.int16) - cfg.neg_decay, 0).astype(np.uint8)
            # ---- obstacle surface ----
            o = kk == KIND_OBST
            if o.any():
                order, cells, starts = _group(f[o])
                mx = np.maximum.reduceat(zz[o][order], starts)
                zt = P["zt"].reshape(-1); lo = P["lo"].reshape(-1)
                zt[cells] = np.fmax(zt[cells], mx)
                lo[cells] = np.minimum(lo[cells].astype(np.int16) + cfg.lo_hit, cfg.lo_max).astype(np.int8)
                P["stamp"].reshape(-1)[cells] = stamp
            # ---- overhead surface ----
            v = kk == KIND_OVER
            if v.any():
                order, cells, starts = _group(f[v])
                mn = np.minimum.reduceat(zz[v][order], starts)
                zc = P["zc"].reshape(-1)
                zc[cells] = np.fmin(zc[cells], mn)
            # ---- class evidence (overhead returns do not vote: class describes what blocks) ----
            h = (cc >= 0) & (kk != KIND_OVER)
            if h.any():
                lin, cnt = np.unique(f[h] * K + cc[h], return_counts=True)
                hist = P["hist"].reshape(-1, K)
                cu = np.unique(lin // K)
                hc = hist[cu]
                big = hc.max(1) > 180
                hc[big] >>= 1
                hist[cu] = hc
                hf = hist.reshape(-1)
                hf[lin] = np.minimum(hf[lin].astype(np.int32) + np.minimum(cnt, 60), 255).astype(np.uint8)
        return ok, lvl, flat

    def add_negative(self, x, y, amount):
        ok, tid, lvl, flat = self.locate(x, y)
        for L in self.levels:
            sel = ok & (lvl == L)
            if sel.any():
                neg = self.pools[L].a["neg"].reshape(-1)
                order, cells, starts = _group(flat[sel])
                add = np.maximum.reduceat(amount[sel][order], starts)
                neg[cells] = np.minimum(neg[cells].astype(np.float32) + add, 255).astype(np.uint8)

    def mark_dynamic(self, x, y, frame: int):
        ok, tid, lvl, flat = self.locate(x, y)
        for L in self.levels:
            sel = ok & (lvl == L)
            if sel.any():
                self.pools[L].a["dyn"].reshape(-1)[flat[sel]] = 255

    # ---------------------------------------------------------------- export
    def tile_ids(self):
        return np.flatnonzero(self.alive)

    def n_cells(self) -> int:
        ids = self.tile_ids()
        return int(((self.T >> self.tlevel[ids].astype(np.int64)) ** 2).sum())

    def memory_bytes(self) -> int:
        """Payload a deployed map holds: live cells x 27 B + 16 B of bookkeeping per tile."""
        return self.n_cells() * BYTES_PER_CELL + len(self.tile_ids()) * TILE_OVERHEAD

    def level_counts(self):
        ids = self.tile_ids()
        tiles = np.bincount(self.tlevel[ids], minlength=len(self.levels))
        cells = tiles * np.array([(self.T >> L) ** 2 for L in self.levels])
        return tiles, cells

    def cells(self, observed_only: bool = True, fields=("zg", "zt", "zc", "lo", "dyn", "neg", "ng", "stamp")):
        """Flatten the live map into per-cell arrays (centres in world metres)."""
        cfg = self.cfg
        ids = self.tile_ids()
        out = {k: [] for k in ("x", "y", "level", "cls", "conf", "tile", "pool_flat") + tuple(fields)}
        for L in self.levels:
            t = ids[self.tlevel[ids] == L]
            if len(t) == 0:
                continue
            P = self.pools[L].a
            slots = self.tslot[t]
            n = self.T >> L; ncell = n * n
            res = cfg.level_res(L)
            lo = P["lo"][slots]; ng = P["ng"][slots]
            if observed_only:
                m = (ng > 0) | (lo > 0) | np.isfinite(P["zc"][slots]) | (P["neg"][slots] > 0)
            else:
                m = np.ones(lo.shape, bool)
            ti, ci = np.nonzero(m)
            ix, iy = self.inv[L]
            out["x"].append(((self.tx[t][ti].astype(np.float64) * n + ix[ci]) + 0.5) * res)
            out["y"].append(((self.ty[t][ti].astype(np.float64) * n + iy[ci]) + 0.5) * res)
            out["level"].append(np.full(len(ti), L, np.int8))
            out["tile"].append(t[ti])
            out["pool_flat"].append(slots[ti] * ncell + ci)
            hist = P["hist"][slots][ti, ci].astype(np.int32)
            hg, ho = hist[:, :2], hist[:, 2:]
            tg, to = hg.sum(1), ho.sum(1)
            use_o = (to > 0) & ((lo[ti, ci] > 0) | (tg == 0))
            cls = np.where(use_o, 2 + ho.argmax(1), np.where(tg > 0, hg.argmax(1), -1)).astype(np.int8)
            conf = np.where(use_o, ho.max(1) / np.maximum(to, 1), np.where(tg > 0, hg.max(1) / np.maximum(tg, 1), 0.0))
            out["cls"].append(cls)
            out["conf"].append(conf.astype(np.float32))
            for k in fields:
                out[k].append(P[k][slots][ti, ci])
        return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}

    # ------------------------------------------------------- per-level access
    def level_view(self, L: int):
        """(tile ids, slots, pool arrays) of every live tile at level L."""
        ids = self.tile_ids()
        t = ids[self.tlevel[ids] == L]
        return t, self.tslot[t], self.pools[L].a

    def cell_xy(self, L: int, t: np.ndarray, ci: np.ndarray):
        """World centre of Z-index `ci` inside tiles `t` (arrays of equal length) at level L."""
        n = self.T >> L
        res = self.cfg.level_res(L)
        ix, iy = self.inv[L]
        return ((self.tx[t].astype(np.float64) * n + ix[ci]) + 0.5) * res, \
               ((self.ty[t].astype(np.float64) * n + iy[ci]) + 0.5) * res

    def tile_summary(self):
        """Per-tile content flags used by the risk policy."""
        cap = len(self.tx)
        occ = np.zeros(cap, bool); neg = np.zeros(cap, np.uint8); dyn = np.zeros(cap, bool)
        for L in self.levels:
            t, slots, P = self.level_view(L)
            if len(t) == 0:
                continue
            lo = P["lo"][slots]
            occ[t] = (lo > 0).any(1)
            neg[t] = P["neg"][slots].max(1)
            dyn[t] = ((P["dyn"][slots] > 0) & (lo > 0)).any(1)
        return occ, neg, dyn

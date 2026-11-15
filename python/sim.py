"""
TERRA-FOVEA :: procedural LiDAR simulator
=========================================
Ray-casts a 64-beam spinning LiDAR through a procedural scene and writes the result
in SemanticKITTI layout, so the exact same loader/pipeline runs on real data:

    <out>/velodyne/000000.bin   float32 [x y z intensity] in the sensor frame
    <out>/labels/000000.label   uint32  (lower 16 bits = SemanticKITTI class id)
    <out>/poses.txt             3x4 sensor->world pose per frame (calib is identity)
    <out>/times.txt

The scene contains everything the problem statement talks about: drivable road, rough
non-drivable verges, kerbs, potholes and trenches (negative obstacles), walls, poles,
vegetation, overhead structures (passable gantry, non-passable boom), parked vehicles,
moving vehicles and pedestrians.

This is SYNTHETIC data. It exists so the whole stack can be developed, unit-tested and
demoed offline; numbers measured on it must always be reported as "simulated".
"""
import argparse
import os
import numpy as np

# SemanticKITTI ids used by the simulator
ROAD, TERRAIN, OTHER_GROUND, BUILDING, STRUCT = 40, 72, 49, 50, 52
POLE, VEG, TRUNK, CAR, PERSON = 80, 70, 71, 10, 30
MOVING_CAR, MOVING_PERSON = 252, 254
NEGATIVE = 201          # simulator-only id: pothole / trench surface (maps to non-drivable)

ROAD_HALF = 5.0         # road spans |y| < 5 m
SENSOR_H = 1.8


# --------------------------------------------------------------------------- scene
class Scene:
    def __init__(self):
        self.boxes = []      # dict(c=(x,y), yaw, l, w, z0, z1, label, traj)
        self.cyls = []       # dict(c=(x,y), r, z0, z1, label, traj)
        self.ells = []       # dict(c=(x,y,z), r=(rx,ry,rz), label)
        self.holes = []      # dict(kind='pothole'|'trench', ...)
        self.curbs = []      # (side(+1/-1), x0, x1)
        self.ego_wp = [(0.0, -1.75)]   # (x, y) waypoints of the ego path
        self.ego_speed = [(0.0, 6.0)]  # (x, speed) profile
        self.seed = 0

    # -- helpers to add things
    def box(self, x, y, l, w, z0, z1, label, yaw=0.0, traj=None):
        self.boxes.append(dict(c=(x, y), yaw=yaw, l=l, w=w, z0=z0, z1=z1, label=label, traj=traj))

    def car(self, x, y, yaw=0.0, traj=None, moving=False):
        lab = MOVING_CAR if moving else CAR
        self.box(x, y, 4.3, 1.8, 0.22, 0.95, lab, yaw, traj)
        self.boxes.append(dict(c=(x, y), yaw=yaw, l=2.3, w=1.6, z0=0.95, z1=1.5, label=lab,
                               traj=traj, off=(-0.25, 0.0)))

    def person(self, x, y, traj=None, moving=False):
        self.cyls.append(dict(c=(x, y), r=0.23, z0=0.0, z1=1.72,
                              label=MOVING_PERSON if moving else PERSON, traj=traj))

    def pole(self, x, y, h=5.0, r=0.09):
        self.cyls.append(dict(c=(x, y), r=r, z0=0.0, z1=h, label=POLE, traj=None))

    def tree(self, x, y, rng):
        h = rng.uniform(3.2, 4.5)
        self.cyls.append(dict(c=(x, y), r=rng.uniform(0.15, 0.25), z0=0.0, z1=h, label=TRUNK, traj=None))
        self.ells.append(dict(c=(x, y, h + 1.2), r=(rng.uniform(1.8, 2.8), rng.uniform(1.8, 2.8), 1.6), label=VEG))

    def bush(self, x, y, rng):
        r = rng.uniform(0.5, 0.9)
        self.ells.append(dict(c=(x, y, r * 0.6), r=(r, r, r * 0.8), label=VEG))

    # -- terrain
    def base(self, x, y):
        return 0.35 * np.sin(x / 38.0) + 0.12 * np.sin(y / 21.0 + 1.0)

    def curb_raise(self, x, y):
        out = np.zeros_like(x)
        for side, x0, x1 in self.curbs:
            d = side * y - ROAD_HALF
            m = (x >= x0) & (x <= x1)
            out = np.where(m, np.maximum(out, 0.15 * np.clip(d / 0.03, 0.0, 1.0)), out)
        return out

    def rough(self, x, y):
        on_verge = np.clip((np.abs(y) - ROAD_HALF) / 0.3, 0.0, 1.0)
        r = 0.035 * (np.sin(7.3 * x + 1.3 * y) * np.sin(5.1 * y - 2.2 * x) + np.sin(13.7 * x) * np.sin(11.9 * y))
        r += 0.02 * np.sin(23.0 * x + 17.0 * y)
        return on_verge * r

    def hole_depth(self, x, y):
        d = np.zeros_like(x)
        for h in self.holes:
            if h['kind'] == 'pothole':
                rr = np.hypot((x - h['x']) / h['rx'], (y - h['y']) / h['ry'])
                d = np.maximum(d, h['depth'] * np.clip((1.0 - rr) / 0.25, 0.0, 1.0))
            else:
                ex = np.clip((h['wx'] / 2 - np.abs(x - h['x'])) / 0.04, 0.0, 1.0)
                ey = np.clip((h['wy'] / 2 - np.abs(y - h['y'])) / 0.04, 0.0, 1.0)
                d = np.maximum(d, h['depth'] * np.minimum(ex, ey))
        return d

    def height(self, x, y):
        return self.base(x, y) + self.curb_raise(x, y) + self.rough(x, y) - self.hole_depth(x, y)

    def support(self, x, y):
        """ground height that objects stand on"""
        x = np.asarray(x, float); y = np.asarray(y, float)
        return self.base(x, y) + self.curb_raise(x, y)

    def ground_label(self, x, y):
        lab = np.full(x.shape, ROAD, np.uint32)
        lab[np.abs(y) >= ROAD_HALF] = TERRAIN
        cr = self.curb_raise(x, y)
        d = np.abs(y) - ROAD_HALF
        lab[(cr > 0.0) & (d < 0.30)] = OTHER_GROUND
        lab[self.hole_depth(x, y) > 0.04] = NEGATIVE
        return lab


def _traj_pos(o, t):
    tr = o.get('traj')
    if tr is None:
        return o['c'][0], o['c'][1], o.get('yaw', 0.0), True
    t0, t1 = tr.get('t0', -1e9), tr.get('t1', 1e9)
    tc = min(max(t, t0), t1) - tr.get('tref', 0.0)
    x = tr['p0'][0] + tr['v'][0] * tc
    y = tr['p0'][1] + tr['v'][1] * tc
    yaw = np.arctan2(tr['v'][1], tr['v'][0]) if (tr['v'][0] or tr['v'][1]) else o.get('yaw', 0.0)
    return x, y, yaw, True


# --------------------------------------------------------------------------- lidar
class Lidar:
    def __init__(self, n_az=1440, n_beams=64, rmax=100.0, rmin=2.7):
        el = np.deg2rad(np.linspace(-24.8, 2.0, n_beams))
        az = np.linspace(-np.pi, np.pi, n_az, endpoint=False)
        A, E = np.meshgrid(az, el)
        self.dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        self.rmax, self.rmin = rmax, rmin

    def scan(self, scene, pos, yaw, t, rng):
        c, s = np.cos(yaw), np.sin(yaw)
        d = self.dirs
        dw = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]], 1)
        n = len(dw)
        T = np.full(n, np.inf)
        L = np.zeros(n, np.uint32)
        ox, oy, oz = pos

        # ---- terrain (height field with kerbs/holes): window search around the smooth hit
        down = dw[:, 2] < -0.004
        idx = np.nonzero(down)[0]
        dz = dw[idx, 2]
        t0 = (oz - scene.base(np.full(len(idx), ox), np.full(len(idx), oy))) / (-dz)
        for _ in range(4):
            t0 = (oz - scene.base(ox + t0 * dw[idx, 0], oy + t0 * dw[idx, 1])) / (-dz)
        win = np.clip(0.75 / (-dz), 0.4, 45.0)
        K = 56
        fr = np.linspace(-1.0, 1.0, K)
        ts = np.maximum(t0[:, None] + win[:, None] * fr[None, :], 0.05)
        X = ox + ts * dw[idx, 0:1]; Y = oy + ts * dw[idx, 1:2]
        F = (oz + ts * dz[:, None]) - scene.height(X, Y)
        neg = F <= 0
        first = np.argmax(neg, 1)
        has = neg.any(1)
        k1 = np.clip(first, 1, K - 1); r = np.arange(len(idx))
        ta, tb = ts[r, k1 - 1], ts[r, k1]
        for _ in range(7):                                   # bisection refine
            tm = 0.5 * (ta + tb)
            fm = (oz + tm * dz) - scene.height(ox + tm * dw[idx, 0], oy + tm * dw[idx, 1])
            below = fm <= 0
            tb = np.where(below, tm, tb); ta = np.where(below, ta, tm)
        tg = np.where(has, np.where(first == 0, ts[:, 0], 0.5 * (ta + tb)), np.inf)
        T[idx] = tg
        gx = ox + tg * dw[idx, 0]; gy = oy + tg * dw[idx, 1]
        ok = np.isfinite(tg)
        gl = np.zeros(len(idx), np.uint32)
        gl[ok] = scene.ground_label(gx[ok], gy[ok])
        L[idx] = gl

        def hit(tn, lab):
            nonlocal T, L
            m = tn < T
            T = np.where(m, tn, T); L = np.where(m, np.uint32(lab), L)

        # ---- oriented boxes
        for b in scene.boxes:
            bx, by, byaw, _ = _traj_pos(b, t)
            cb, sb = np.cos(byaw), np.sin(byaw)
            off = b.get('off', (0.0, 0.0))
            bx += cb * off[0] - sb * off[1]; by += sb * off[0] + cb * off[1]
            if np.hypot(bx - ox, by - oy) > self.rmax + 12 + b['l']:
                continue
            zb = float(scene.support(bx, by)) if b.get('ground', True) else 0.0
            px, py = ox - bx, oy - by
            lo = np.array([cb * px + sb * py, -sb * px + cb * py, oz - zb])
            ld = np.stack([cb * dw[:, 0] + sb * dw[:, 1], -sb * dw[:, 0] + cb * dw[:, 1], dw[:, 2]], 1)
            mn = np.array([-b['l'] / 2, -b['w'] / 2, b['z0']]); mx = np.array([b['l'] / 2, b['w'] / 2, b['z1']])
            with np.errstate(divide='ignore', invalid='ignore'):
                inv = 1.0 / ld
                t1 = (mn - lo) * inv; t2 = (mx - lo) * inv
            tn = np.nanmax(np.minimum(t1, t2), 1); tf = np.nanmin(np.maximum(t1, t2), 1)
            tn = np.where((tn < tf) & (tn > 0.05), tn, np.inf)
            hit(tn, b['label'])

        # ---- vertical cylinders
        for cy in scene.cyls:
            cx, cyy, _, _ = _traj_pos(cy, t)
            if np.hypot(cx - ox, cyy - oy) > self.rmax + 2:
                continue
            zb = float(scene.support(cx, cyy))
            px, py = ox - cx, oy - cyy
            a = dw[:, 0] ** 2 + dw[:, 1] ** 2
            bq = 2 * (px * dw[:, 0] + py * dw[:, 1])
            cq = px * px + py * py - cy['r'] ** 2
            disc = bq * bq - 4 * a * cq
            with np.errstate(invalid='ignore', divide='ignore'):
                tn = (-bq - np.sqrt(disc)) / (2 * a)
            z = oz + tn * dw[:, 2] - zb
            tn = np.where((disc > 0) & (tn > 0.05) & (z >= cy['z0']) & (z <= cy['z1']), tn, np.inf)
            hit(tn, cy['label'])

        # ---- ellipsoids (canopies, bushes)
        for e in scene.ells:
            ex, ey, ez = e['c']
            if np.hypot(ex - ox, ey - oy) > self.rmax + 4:
                continue
            zb = float(scene.support(ex, ey))
            rr = np.array(e['r'])
            p = np.array([ox - ex, oy - ey, oz - ez - zb]) / rr
            q = dw / rr
            a = (q * q).sum(1); bq = 2 * (q @ p); cq = p @ p - 1.0
            disc = bq * bq - 4 * a * cq
            with np.errstate(invalid='ignore'):
                tn = (-bq - np.sqrt(disc)) / (2 * a)
            tn = np.where((disc > 0) & (tn > 0.05), tn, np.inf)
            hit(tn, e['label'])

        # ---- sensor model: noise, dropout, range gate
        valid = np.isfinite(T) & (T < self.rmax) & (T > self.rmin)
        veg = (L == VEG)
        sigma = np.where(veg, 0.05, np.where(L == TERRAIN, 0.02, 0.012))
        Tn = T + rng.normal(0, 1, n) * sigma
        drop = rng.random(n) < np.where(veg, 0.30, 0.03)
        valid &= ~drop
        pts_s = d[valid] * Tn[valid, None]                    # sensor frame
        lab = L[valid]
        inten = np.zeros(len(lab), np.float32)
        table = {ROAD: (0.25, 0.03), TERRAIN: (0.45, 0.08), OTHER_GROUND: (0.50, 0.05), NEGATIVE: (0.22, 0.05),
                 BUILDING: (0.55, 0.06), STRUCT: (0.60, 0.06), POLE: (0.80, 0.08), VEG: (0.35, 0.12),
                 TRUNK: (0.30, 0.08), CAR: (0.65, 0.15), MOVING_CAR: (0.65, 0.15),
                 PERSON: (0.40, 0.10), MOVING_PERSON: (0.40, 0.10)}
        for k, (mu, sg) in table.items():
            m = lab == k
            inten[m] = rng.normal(mu, sg, int(m.sum()))
        inten = np.clip(inten, 0.0, 1.0)
        return np.concatenate([pts_s, inten[:, None]], 1).astype(np.float32), lab.astype(np.uint32)


# --------------------------------------------------------------------------- scenarios
def _street_furniture(sc, rng, x0, x1):
    """buildings, poles, trees, bushes along the road"""
    x = x0
    while x < x1:                                             # buildings both sides
        for side in (1, -1):
            if rng.random() < 0.75:
                l = rng.uniform(8, 18); dep = rng.uniform(5, 9)
                yy = side * (rng.uniform(10.5, 14.0) + dep / 2)
                sc.box(x + l / 2, yy, l, dep, 0.0, rng.uniform(4, 9), BUILDING)
        x += rng.uniform(16, 24)
    for x in np.arange(x0 + 6, x1, 22.0):
        sc.pole(x, 5.9); sc.pole(x + 11.0, -5.9)
    x = x0 + 3
    while x < x1:
        side = rng.choice([1, -1])
        if rng.random() < 0.6:
            sc.tree(x, side * rng.uniform(7.2, 9.0), rng)
        else:
            sc.bush(x, side * rng.uniform(6.3, 9.0), rng)
        x += rng.uniform(5, 11)


def gauntlet():
    """The demo / evaluation course. Every PS requirement appears at a known place."""
    sc = Scene(); rng = np.random.default_rng(7); sc.seed = 7
    _street_furniture(sc, rng, -30, 235)
    sc.curbs = [(1, -30, 62), (-1, 48, 140), (1, 100, 235)]
    # parked vehicles in the right-hand bay
    for x in (16.0, 22.5, 58.0, 132.0):
        sc.car(x, -4.0)
    # A  pothole in the ego lane                      (negative obstacle, x=30)
    sc.holes.append(dict(kind='pothole', x=30.0, y=-1.6, rx=0.55, ry=0.5, depth=0.20))
    # B  pedestrian crossing the road                 (dynamic + path relevance, x=46)
    sc.person(46.0, 7.0, traj=dict(p0=(46.0, 7.0), v=(0.0, -1.4), t0=2.5, t1=12.5, tref=2.5), moving=True)
    # C  oncoming vehicle in the other lane
    sc.car(150.0, 1.9, traj=dict(p0=(150.0, 1.9), v=(-8.0, 0.0), t0=0.0, t1=40.0, tref=0.0), moving=True)
    # D  gantry over the road: 3.4 m clearance -> passable, classic 2D grids block it (x=70)
    sc.box(70.0, 0.0, 1.2, 12.6, 3.4, 3.9, STRUCT)
    sc.pole(70.0, 6.2, h=3.9, r=0.18); sc.pole(70.0, -6.2, h=3.9, r=0.18)
    # E  trench across the ego lane                   (negative obstacle, x=86)
    sc.holes.append(dict(kind='trench', x=86.0, y=-2.6, wx=0.8, wy=4.6, depth=0.55))
    # F  low boom over the oncoming lane: 1.35 m clearance -> NOT passable (x=97)
    sc.box(97.0, 2.6, 0.12, 4.8, 1.35, 1.47, STRUCT)
    sc.pole(97.0, 5.2, h=1.5, r=0.10)
    # G  vehicle crossing from a side street (x=112)
    sc.car(112.0, -40.0, traj=dict(p0=(112.0, -40.0), v=(0.0, 6.5), t0=15.5, t1=28.0, tref=15.5), moving=True)
    # H  narrow gate: 3.0 m gap between two blocks (x=124)
    sc.box(124.0, -3.9, 1.0, 2.2, 0.0, 1.0, STRUCT); sc.box(124.0, 1.3, 1.0, 2.2, 0.0, 1.0, STRUCT)
    # pedestrian walking along the verge (dynamic, NOT path relevant)
    sc.person(20.0, 6.6, traj=dict(p0=(20.0, 6.6), v=(1.2, 0.0), t0=0.0, t1=60.0, tref=0.0), moving=True)
    sc.person(64.0, -6.4)                                   # standing person (static)
    sc.holes.append(dict(kind='pothole', x=52.0, y=7.5, rx=0.9, ry=0.8, depth=0.35))   # crater on the verge
    # ego path: dodge pothole, yield for pedestrian, pass trench on the other lane, return
    sc.ego_wp = [(0, -1.75), (22, -1.75), (28, -3.0), (33, -3.0), (39, -1.75), (76, -1.75),
                 (83, 1.7), (89, 1.7), (94, -1.3), (118, -1.3), (160, -1.3)]
    sc.ego_speed = [(0, 6.0), (28, 4.0), (33, 2.2), (40, 2.0), (44, 5.5), (74, 6.0), (80, 4.5),
                    (92, 4.5), (104, 3.0), (108, 5.5), (160, 6.0)]
    return sc


def random_scene(seed):
    sc = Scene(); rng = np.random.default_rng(seed); sc.seed = seed
    x1 = 190.0
    _street_furniture(sc, rng, -40, x1)
    for side in (1, -1):
        x = -40.0
        while x < x1:
            ln = rng.uniform(20, 70)
            if rng.random() < 0.6:
                sc.curbs.append((side, x, x + ln))
            x += ln
    for _ in range(rng.integers(3, 7)):
        sc.car(rng.uniform(5, 150), rng.choice([-4.0, 4.0]) + rng.normal(0, 0.15), yaw=rng.normal(0, 0.04))
    for _ in range(rng.integers(2, 4)):
        v = rng.uniform(4, 10); x0 = rng.uniform(40, 170)
        sc.car(x0, 1.9, traj=dict(p0=(x0, 1.9), v=(-v, 0.0), t0=0, t1=99, tref=0), moving=True)
    for _ in range(rng.integers(1, 3)):
        x0 = rng.uniform(25, 120); v = rng.uniform(4, 8) * rng.choice([1, -1])
        sc.car(x0, -30.0 * np.sign(v), traj=dict(p0=(x0, -30.0 * np.sign(v)), v=(0.0, v), t0=0, t1=99, tref=0), moving=True)
    for _ in range(rng.integers(3, 7)):
        x0 = rng.uniform(8, 110); side = rng.choice([1, -1])
        if rng.random() < 0.55:
            v = rng.uniform(0.9, 1.8)
            sc.person(x0, side * 7.0, traj=dict(p0=(x0, side * 7.0), v=(0.0, -side * v), t0=0, t1=99, tref=0), moving=True)
        elif rng.random() < 0.5:
            v = rng.uniform(0.8, 1.6) * rng.choice([1, -1])
            sc.person(x0, side * 6.5, traj=dict(p0=(x0, side * 6.5), v=(v, 0.0), t0=0, t1=99, tref=0), moving=True)
        else:
            sc.person(x0, side * rng.uniform(1.0, 7.0))
    for _ in range(rng.integers(2, 5)):
        if rng.random() < 0.6:
            sc.holes.append(dict(kind='pothole', x=rng.uniform(10, 120), y=rng.uniform(-4, 4),
                                 rx=rng.uniform(0.35, 0.8), ry=rng.uniform(0.35, 0.8), depth=rng.uniform(0.14, 0.35)))
        else:
            sc.holes.append(dict(kind='trench', x=rng.uniform(15, 120), y=rng.uniform(-3, 3),
                                 wx=rng.uniform(0.5, 1.2), wy=rng.uniform(2, 5), depth=rng.uniform(0.3, 0.7)))
    if rng.random() < 0.7:
        gx = rng.uniform(30, 120); zc = rng.uniform(2.9, 4.2)
        sc.box(gx, 0.0, 1.2, 12.6, zc, zc + 0.5, STRUCT); sc.pole(gx, 6.2, h=zc + 0.5, r=0.18); sc.pole(gx, -6.2, h=zc + 0.5, r=0.18)
    if rng.random() < 0.7:
        bx = rng.uniform(30, 120); side = rng.choice([1, -1]); zc = rng.uniform(1.0, 1.7)
        sc.box(bx, side * 2.6, 0.12, 4.8, zc, zc + 0.12, STRUCT); sc.pole(bx, side * 5.2, h=zc + 0.15, r=0.10)
    for _ in range(rng.integers(0, 3)):
        sc.box(rng.uniform(20, 130), rng.uniform(-4, 4), rng.uniform(0.6, 1.5), rng.uniform(0.8, 2.5), 0.0, rng.uniform(0.6, 1.4), STRUCT)
    sc.ego_wp = [(0, -1.75), (200, -1.75)]
    sc.ego_speed = [(0, float(rng.uniform(4, 8))), (200, float(rng.uniform(4, 8)))]
    return sc


# --------------------------------------------------------------------------- driver
def ego_states(sc, n_frames, dt=0.1):
    wx = np.array([w[0] for w in sc.ego_wp]); wy = np.array([w[1] for w in sc.ego_wp])
    sx = np.array([s[0] for s in sc.ego_speed]); sv = np.array([s[1] for s in sc.ego_speed])

    def ypath(x):
        i = np.clip(np.searchsorted(wx, x) - 1, 0, len(wx) - 2)
        u = np.clip((x - wx[i]) / (wx[i + 1] - wx[i]), 0, 1)
        u = u * u * (3 - 2 * u)
        return wy[i] + (wy[i + 1] - wy[i]) * u
    out = []; x = 0.0
    for k in range(n_frames):
        y = float(ypath(x)); yaw = float(np.arctan2(ypath(x + 0.5) - ypath(x - 0.5), 1.0))
        z = float(sc.support(x, y)) + SENSOR_H
        out.append((k * dt, x, y, z, yaw))
        x += float(np.interp(x, sx, sv)) * dt * np.cos(yaw)
    return out


def generate(sc, out, n_frames, stride=1, lidar=None, quiet=False):
    os.makedirs(os.path.join(out, 'velodyne'), exist_ok=True)
    os.makedirs(os.path.join(out, 'labels'), exist_ok=True)
    lidar = lidar or Lidar()
    rng = np.random.default_rng(sc.seed + 1000)
    states = ego_states(sc, n_frames * stride)[::stride]
    with open(os.path.join(out, 'poses.txt'), 'w') as fp, open(os.path.join(out, 'times.txt'), 'w') as ft:
        for i, (t, x, y, z, yaw) in enumerate(states):
            pts, lab = lidar.scan(sc, (x, y, z), yaw, t, rng)
            pts.tofile(os.path.join(out, 'velodyne', f'{i:06d}.bin'))
            lab.tofile(os.path.join(out, 'labels', f'{i:06d}.label'))
            c, s = np.cos(yaw), np.sin(yaw)
            fp.write(' '.join(f'{v:.6f}' for v in [c, -s, 0, x, s, c, 0, y, 0, 0, 1, z]) + '\n')
            ft.write(f'{t:.3f}\n')
            if not quiet and i % 20 == 0:
                print(f'  frame {i}/{len(states)}  {len(pts)} pts', flush=True)
    with open(os.path.join(out, 'calib.txt'), 'w') as f:
        f.write('Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n')
    with open(os.path.join(out, 'SIMULATED.txt'), 'w') as f:
        f.write('Synthetic sequence produced by python/sim.py. Not real sensor data.\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data')
    ap.add_argument('--frames', type=int, default=240)
    ap.add_argument('--train-scenes', type=int, default=8)
    ap.add_argument('--train-frames', type=int, default=10)
    a = ap.parse_args()
    print('gauntlet course ->', os.path.join(a.out, 'gauntlet'))
    generate(gauntlet(), os.path.join(a.out, 'gauntlet'), a.frames)
    for s in range(a.train_scenes):
        d = os.path.join(a.out, f'train_{s:02d}')
        print('training scene ->', d)
        generate(random_scene(100 + s), d, a.train_frames, stride=12)
    for s in range(2):
        d = os.path.join(a.out, f'val_{s:02d}')
        print('validation scene ->', d)
        generate(random_scene(900 + s), d, 60, stride=1)

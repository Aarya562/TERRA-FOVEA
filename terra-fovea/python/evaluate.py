"""
Evaluation harness: every number we quote comes out of this script.

    python python/evaluate.py --seq data/gauntlet

Runs the same sequence through several configurations of the same engine and measures
  memory        allocated cells / bytes, against uniform grids
  latency       per-stage milliseconds and FPS
  segmentation  per-class IoU, and accuracy split by distance band
  map quality   obstacle footprint precision / recall against 5 cm ground truth (whole map and
                the safety-critical region: the driving corridor)
  ghosts        occupied cells left behind by moving objects (ray clearing on / off)
  hazards       negative-obstacle detection range, false alarms, overhead clearance error
Writes results/metrics.json, results/REPORT.md and charts. Needs point-wise labels (.label files).
"""
import argparse
import json
import os
import time
import numpy as np
from engine import Engine, Sequence, LUT, MOVING, ROOT, CLASS_NAMES, P_DYNAMIC, P_AI, C_NEGATIVE, C_DYNAMIC

BANDS = [(0, 10), (10, 20), (20, 40), (40, 100)]
VARIANTS = [
    ('uniform_5cm', 'policy=0;uniform_level=0', 'Uniform 5 cm (sparse)'),
    ('uniform_40cm', 'policy=0;uniform_level=3', 'Uniform 40 cm'),
    ('distance', 'policy=1', 'Distance-only foveation'),
    ('risk', 'policy=2', 'TERRA-FOVEA (risk-adaptive)'),
    ('risk_no_rayclear', 'policy=2;ray_clear=0', 'risk-adaptive, ray clearing OFF'),
    ('risk_full_ai', 'policy=2;selective_ai=0', 'risk-adaptive, network on ALL points'),
]
Q = 0.05
SH = 1 << 20


def keys(x, y):
    return (np.floor(x / Q).astype(np.int64) + SH) * (2 * SH) + (np.floor(y / Q).astype(np.int64) + SH)


def dilate(k, r=1):
    o = [k + dx * (2 * SH) + dy for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
    return np.unique(np.concatenate(o))


def cell_keys(cells):
    """expand variable-size cells to 5 cm keys"""
    out = []
    for lvl in np.unique(cells[:, 11]).astype(int):
        c = cells[cells[:, 11] == lvl]; k = 1 << lvl; s = Q * k
        ix = np.round((c[:, 0] - s / 2) / Q).astype(np.int64); iy = np.round((c[:, 1] - s / 2) / Q).astype(np.int64)
        dx, dy = np.meshgrid(np.arange(k), np.arange(k))
        X = (ix[:, None] + dx.ravel()[None]).ravel(); Y = (iy[:, None] + dy.ravel()[None]).ravel()
        out.append((X + SH) * (2 * SH) + (Y + SH))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)


def key_xy(k):
    return ((k // (2 * SH)) - SH + 0.5) * Q, ((k % (2 * SH)) - SH + 0.5) * Q


def run(seq, cfg, every=4, perturb=None, max_frames=None, scene=None):
    eng = Engine(cfg)
    n = len(seq) if not max_frames else min(len(seq), max_frames)
    conf = np.zeros((8, 8), np.int64); band_ok = np.zeros(len(BANDS)); band_n = np.zeros(len(BANDS))
    band_conf = np.zeros((len(BANDS), 8, 8), np.int64)
    dyn = np.zeros(3)                                    # tp fp fn (points)
    st_sum = {}; st_peak = {}; nst = 0
    g_static = np.zeros(0, np.int64); g_moved = np.zeros(0, np.int64)
    mq = {k: [] for k in ('prec', 'rec', 'iou', 'cprec', 'crec', 'ciou', 'cfalse_m2', 'ghost_m2', 'occ_cells')}
    decisions = []; hazards = {}; series = []
    rng = np.random.default_rng(0)
    for i in range(n):
        pts = seq.scan(i); lab = seq.labels(i); pose = seq.poses[i]
        if perturb:
            keep = rng.random(len(pts)) > perturb['dropout']
            pts = pts[keep].copy(); lab = lab[keep]
            r = np.linalg.norm(pts[:, :3], axis=1, keepdims=True)
            pts[:, :3] *= 1 + rng.normal(0, perturb['sigma'], (len(pts), 1)) / np.maximum(r, 1)
        eng.process(pts, pose, seq.times[i])
        s = eng.stats()
        if i >= 5:
            nst += 1
            for k, v in s.items():
                st_sum[k] = st_sum.get(k, 0) + v; st_peak[k] = max(st_peak.get(k, 0), v)
        d = eng.decision()
        if not decisions or decisions[-1][1] != d['state'] or decisions[-1][3] != d['kind']:
            decisions.append((i, d['state'], d['reason'], d['kind'], round(float(pose[0, 3]), 1)))
        series.append((i, s['map_bytes'], s['cells_alloc'], s['ms_total'], s['ai_points'] / max(1, s['points'])))
        # ---- point-wise segmentation
        cls, fl, _ = eng.point_labels(); y = LUT[lab]; rr = np.linalg.norm(pts[:, :3], axis=1)
        m = (y > 0) & (rr <= 100)
        np.add.at(conf, (y[m], cls[m]), 1)
        for b, (r0, r1) in enumerate(BANDS):
            mb = m & (rr >= r0) & (rr < r1)
            band_ok[b] += (cls[mb] == y[mb]).sum(); band_n[b] += mb.sum()
            np.add.at(band_conf[b], (y[mb], cls[mb]), 1)
        gm = MOVING[lab] & m; pd = (fl & P_DYNAMIC) > 0
        dyn += [(gm & pd).sum(), (~gm & pd & m).sum(), (gm & ~pd).sum()]
        # ---- ground truth occupancy at 5 cm (accumulated static surface + current movers)
        w = pts[:, :3] @ pose[:3, :3].T + pose[:3, 3]
        obst = (y >= 3) & (pts[:, 2] < 0.7) & (rr <= 100)
        ks = keys(w[obst & ~MOVING[lab], 0], w[obst & ~MOVING[lab], 1])
        km = keys(w[obst & MOVING[lab], 0], w[obst & MOVING[lab], 1])
        g_static = np.union1d(g_static, ks)
        if i % every == every - 1 or i == n - 1:
            cells = eng.cells(); ex, ey = pose[0, 3], pose[1, 3]; yaw = np.arctan2(pose[1, 0], pose[0, 0])
            occ = cells[(cells[:, 4] > 0.2) & ((cells[:, 8] >= 0.5) | ((cells[:, 9].astype(int) & C_DYNAMIC) > 0))]
            M = cell_keys(occ); G = np.union1d(g_static, km)
            mx, my = key_xy(M); gx, gy = key_xy(G)
            Mr = M[np.hypot(mx - ex, my - ey) < 60]; Gr = G[np.hypot(gx - ex, gy - ey) < 60]
            Gd = dilate(Gr, 1); Md = dilate(Mr, 1)
            tp_m = np.isin(Mr, Gd).sum(); tp_g = np.isin(Gr, Md).sum()
            mq['prec'].append(tp_m / max(1, len(Mr))); mq['rec'].append(tp_g / max(1, len(Gr)))
            mq['iou'].append(tp_m / max(1, len(Mr) + len(Gr) - tp_g)); mq['occ_cells'].append(len(occ))

            def in_corr(k):
                x, yv = key_xy(k); fx = np.cos(yaw) * (x - ex) + np.sin(yaw) * (yv - ey); fy = -np.sin(yaw) * (x - ex) + np.cos(yaw) * (yv - ey)
                return (fx > 2.5) & (fx < 40) & (np.abs(fy) < 2.5)
            Mc = M[in_corr(M)]; Gc = G[in_corr(G)]; Gcd = dilate(Gc, 1)
            tpc = np.isin(Mc, Gcd).sum(); tgc = np.isin(Gc, dilate(Mc, 1)).sum()
            if len(Gc) > 20:
                mq['cprec'].append(tpc / max(1, len(Mc))); mq['crec'].append(tgc / max(1, len(Gc)))
                mq['ciou'].append(tpc / max(1, len(Mc) + len(Gc) - tgc))
            mq['cfalse_m2'].append((len(Mc) - tpc) * Q * Q)
            # ghosts: occupied map cells where a mover used to be and nothing is now
            zone = np.setdiff1d(g_moved, dilate(km, 6)); zone = np.setdiff1d(zone, dilate(g_static, 2))
            mq['ghost_m2'].append(np.isin(M, zone).sum() * Q * Q)
            # hazards against the scene description (simulated sequences only)
            if scene is not None:
                neg = cells[(cells[:, 9].astype(int) & C_NEGATIVE) > 0]
                for hi, h in enumerate(scene.holes):
                    hw = (h['rx'], h['ry']) if h['kind'] == 'pothole' else (h['wx'] / 2, h['wy'] / 2)
                    inside = (np.abs(neg[:, 0] - h['x']) < hw[0] + 0.3) & (np.abs(neg[:, 1] - h['y']) < hw[1] + 0.3)
                    dist = float(np.hypot(h['x'] - ex, h['y'] - ey))
                    rec = hazards.setdefault(f'hole{hi}', dict(kind=h['kind'], x=h['x'], y=h['y'], depth=h['depth'], detect_dist=None, min_dist=1e9))
                    rec['min_dist'] = min(rec['min_dist'], dist)
                    if inside.sum() >= 3 and rec['detect_dist'] is None and h['x'] > ex:
                        rec['detect_dist'] = round(dist, 1)
                far = np.ones(len(neg), bool)
                for h in scene.holes:
                    hw = (h['rx'], h['ry']) if h['kind'] == 'pothole' else (h['wx'] / 2, h['wy'] / 2)
                    far &= ~((np.abs(neg[:, 0] - h['x']) < hw[0] + 0.5) & (np.abs(neg[:, 1] - h['y']) < hw[1] + 0.5))
                hazards.setdefault('neg_false_m2', []).append(float((neg[far, 2] ** 2).sum()))
                for name, bx, y0, y1, true_c in (('gantry', 70.0, -4.5, 4.5, 3.4), ('boom', 97.0, 0.6, 4.6, 1.35)):
                    if abs(ex - bx) > 30:
                        continue
                    u = cells[(np.abs(cells[:, 0] - bx) < 0.5) & (cells[:, 1] > y0) & (cells[:, 1] < y1)]
                    if len(u):
                        oc = u[u[:, 5] >= 0]
                        rec = hazards.setdefault(name, dict(true_clearance=true_c, meas=[], blocked_cells=[]))
                        if len(oc):
                            rec['meas'].append(float(np.median(oc[:, 5])))
                        rec['blocked_cells'].append(int(((u[:, 4] > 0.5) & (u[:, 8] >= 0.5)).sum()))
        g_moved = np.union1d(g_moved, km)
    tp = np.diag(conf).astype(float); den = conf.sum(0) + conf.sum(1) - tp
    iou = np.where(den > 0, tp / np.maximum(den, 1), np.nan)
    biou = []
    for b in range(len(BANDS)):
        t = np.diag(band_conf[b]).astype(float); dn = band_conf[b].sum(0) + band_conf[b].sum(1) - t
        biou.append(float(np.nanmean(np.where(dn[1:] > 0, t[1:] / np.maximum(dn[1:], 1), np.nan))))
    out = dict(cfg=cfg, frames=n,
               mean={k: v / max(1, nst) for k, v in st_sum.items()}, peak=st_peak,
               fps=1000.0 / (st_sum['ms_total'] / nst), iou={CLASS_NAMES[k]: float(iou[k]) for k in range(1, 8)},
               miou=float(np.nanmean(iou[1:])), acc=float(tp.sum() / conf.sum()),
               band_acc=(band_ok / np.maximum(band_n, 1)).tolist(), band_miou=biou, band_points=band_n.tolist(),
               dyn_precision=float(dyn[0] / max(1, dyn[0] + dyn[1])), dyn_recall=float(dyn[0] / max(1, dyn[0] + dyn[2])),
               map={k: float(np.mean(v)) if v else None for k, v in mq.items()},
               ghost_peak_m2=float(np.max(mq['ghost_m2'])) if mq['ghost_m2'] else 0.0,
               decisions=decisions, series=series)
    if scene is not None:
        hz = {}
        for k, v in hazards.items():
            if k == 'neg_false_m2':
                hz[k] = float(np.mean(v))
            elif 'meas' in v:
                hz[k] = dict(true_clearance=v['true_clearance'], measured_clearance=float(np.median(v['meas'])) if v['meas'] else None,
                             max_blocking_cells=int(max(v['blocked_cells'])) if v['blocked_cells'] else 0)
            else:
                hz[k] = v
        out['hazards'] = hz
    return out


def charts(R, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    col = {'uniform_5cm': '#8a8f98', 'uniform_40cm': '#c4a35a', 'distance': '#4c8dd6', 'risk': '#e4572e'}
    names = {v[0]: v[2] for v in VARIANTS}
    # 1 memory
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    labels = ['Dense 3D voxels\n5 cm, 200x200x4 m', 'Dense uniform 2.5D\n5 cm', 'Sparse uniform 2.5D\n5 cm', 'Distance-only\nfoveation', 'TERRA-FOVEA\nrisk-adaptive']
    vals = [1.28e9, 16e6 * 16, R['uniform_5cm']['mean']['map_bytes'], R['distance']['mean']['map_bytes'], R['risk']['mean']['map_bytes']]
    b = ax.bar(labels, [v / 1048576 for v in vals], color=['#3d4250', '#5b6170', '#8a8f98', '#4c8dd6', '#e4572e'])
    ax.set_yscale('log'); ax.set_ylabel('map memory (MB, log scale)')
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() * 1.15, f'{v / 1048576:,.1f} MB', ha='center', fontsize=9)
    ax.set_title('Map memory on the same drive (measured; dense grids computed)')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'memory.png'), dpi=130); plt.close(fig)
    # 2 pareto
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for k in ('uniform_5cm', 'uniform_40cm', 'distance', 'risk'):
        ax.scatter(R[k]['mean']['map_bytes'] / 1048576, R[k]['map']['ciou'], s=110, color=col[k], zorder=3)
        ax.annotate(names[k], (R[k]['mean']['map_bytes'] / 1048576, R[k]['map']['ciou']), textcoords='offset points', xytext=(8, 6), fontsize=9)
    ax.set_xscale('log'); ax.set_xlabel('map memory (MB, log)'); ax.set_ylabel('obstacle footprint IoU in the driving corridor')
    ax.set_title('Safety-critical map quality vs memory'); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'pareto.png'), dpi=130); plt.close(fig)
    # 3 latency
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    stages = ['ms_ground', 'ms_triage', 'ms_net', 'ms_track', 'ms_risk', 'ms_insert', 'ms_rayclear', 'ms_decision']
    sc = ['#5b6170', '#8a8f98', '#e4572e', '#c4a35a', '#4c8dd6', '#57a773', '#9b6fc3', '#3d4250']
    rows = [('risk', 'TERRA-FOVEA (selective AI)'), ('risk_full_ai', 'network on all points')]
    for j, (k, nm) in enumerate(rows):
        left = 0
        for s, c in zip(stages, sc):
            v = R[k]['mean'][s]; ax.barh(nm, v, left=left, color=c, label=s[3:] if j == 0 else None); left += v
        ax.text(left + 1, j, f'{left:.0f} ms = {1000 / left:.1f} FPS', va='center', fontsize=9)
    ax.set_xlabel('ms per sweep (this machine, CPU only)'); ax.legend(ncol=4, fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.28))
    ax.set_xlim(0, max(R['risk_full_ai']['mean']['ms_total'], R['risk']['mean']['ms_total']) * 1.35)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'latency.png'), dpi=130); plt.close(fig)
    # 4 accuracy by distance
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(len(BANDS)); w = 0.38
    ax.bar(x - w / 2, R['risk']['band_acc'], w, color='#e4572e', label='point accuracy')
    ax.bar(x + w / 2, R['risk']['band_miou'], w, color='#4c8dd6', label='mIoU')
    ax.set_xticks(x); ax.set_xticklabels([f'{a}-{b} m' for a, b in BANDS]); ax.set_ylim(0, 1.05); ax.legend(frameon=False)
    for i, (a, m) in enumerate(zip(R['risk']['band_acc'], R['risk']['band_miou'])):
        ax.text(i - w / 2, a + 0.01, f'{a:.3f}', ha='center', fontsize=8); ax.text(i + w / 2, m + 0.01, f'{m:.3f}', ha='center', fontsize=8)
    ax.set_title('Segmentation quality by distance from the sensor')
    fig.tight_layout(); fig.savefig(os.path.join(outdir, 'accuracy_by_distance.png'), dpi=130); plt.close(fig)


def report(R, outdir, seq):
    r = R['risk']; d = R['distance']; u = R['uniform_5cm']; c = R['uniform_40cm']; nc = R['risk_no_rayclear']; fa = R['risk_full_ai']
    mb = lambda v: f"{v['mean']['map_bytes'] / 1048576:.2f}"
    L = []
    L.append('# TERRA-FOVEA: measured results\n')
    L.append(f"Sequence: `{seq.dir}` ({r['frames']} sweeps, ~{r['mean']['points']:,.0f} points per sweep). "
             + ('**Simulated LiDAR data** (python/sim.py): all accuracy numbers below are on simulation.' if seq.simulated else 'Real sensor data.'))
    L.append('Latency is wall-clock on the machine that ran this script, CPU only. Regenerate with `python python/evaluate.py`.\n')
    L.append('## 1. Memory\n')
    L.append('| Representation | Cells (mean) | Memory | vs TERRA-FOVEA |\n|---|---:|---:|---:|')
    rb = r['mean']['map_bytes']
    L.append(f"| Dense 3D voxel grid, 5 cm, 200 x 200 x 4 m (computed, 1 byte/voxel) | 1,280,000,000 | 1220.70 MB | {1.28e9 / rb:,.0f}x |")
    L.append(f"| Dense uniform 2.5D grid, 5 cm, 200 x 200 m (computed, 16 B/cell) | 16,000,000 | 244.14 MB | {16e6 * 16 / rb:,.1f}x |")
    L.append(f"| Sparse uniform 2.5D, 5 cm (measured, same engine) | {u['mean']['cells_alloc']:,.0f} | {mb(u)} MB | {u['mean']['map_bytes'] / rb:.1f}x |")
    L.append(f"| Distance-only foveation (measured) | {d['mean']['cells_alloc']:,.0f} | {mb(d)} MB | {d['mean']['map_bytes'] / rb:.2f}x |")
    L.append(f"| **TERRA-FOVEA risk-adaptive (measured)** | **{r['mean']['cells_alloc']:,.0f}** | **{mb(r)} MB** | 1x |")
    L.append(f"| Uniform 40 cm (measured) | {c['mean']['cells_alloc']:,.0f} | {mb(c)} MB | {c['mean']['map_bytes'] / rb:.2f}x |\n")
    L.append(f"Peak map memory of TERRA-FOVEA over the drive: {r['peak']['map_bytes'] / 1048576:.2f} MB. Frame workspace (scratch buffers, not the map): {r['mean']['workspace_bytes'] / 1048576:.1f} MB.\n")
    L.append('![memory](memory.png)\n')
    L.append('## 2. Safety-critical map quality vs memory\n')
    L.append('Obstacle footprint of the map compared with 5 cm ground truth, inside the driving corridor (2.5 to 40 m ahead, 5 m wide). '
             '"Falsely blocked" is free corridor area the map marks as occupied.\n')
    L.append('| Variant | Memory | Corridor IoU | Corridor precision | Corridor recall | Falsely blocked | Whole-map IoU (60 m) |\n|---|---:|---:|---:|---:|---:|---:|')
    for k in ('uniform_5cm', 'uniform_40cm', 'distance', 'risk'):
        v = R[k]; m = v['map']
        L.append(f"| {dict((a, b) for a, _, b in VARIANTS)[k]} | {mb(v)} MB | {m['ciou']:.3f} | {m['cprec']:.3f} | {m['crec']:.3f} | {m['cfalse_m2']:.2f} m2 | {m['iou']:.3f} |")
    L.append('\n![pareto](pareto.png)\n')
    L.append('## 3. Segmentation\n')
    L.append('| Class | ' + ' | '.join(CLASS_NAMES[1:]) + ' | mIoU |\n|---|' + '---:|' * 8)
    L.append('| IoU, selective AI | ' + ' | '.join(f"{r['iou'][n]:.3f}" for n in CLASS_NAMES[1:]) + f" | **{r['miou']:.3f}** |")
    L.append('| IoU, network on all points | ' + ' | '.join(f"{fa['iou'][n]:.3f}" for n in CLASS_NAMES[1:]) + f" | {fa['miou']:.3f} |\n")
    L.append('| Distance band | ' + ' | '.join(f'{a}-{b} m' for a, b in BANDS) + ' |\n|---|' + '---:|' * len(BANDS))
    L.append('| Point accuracy | ' + ' | '.join(f'{v:.3f}' for v in r['band_acc']) + ' |')
    L.append('| mIoU | ' + ' | '.join(f'{v:.3f}' for v in r['band_miou']) + ' |')
    L.append('| Points evaluated | ' + ' | '.join(f'{v:,.0f}' for v in r['band_points']) + ' |\n')
    L.append(f"Moving-object points (tracker): precision {r['dyn_precision']:.3f}, recall {r['dyn_recall']:.3f}.\n")
    L.append('![accuracy](accuracy_by_distance.png)\n')
    L.append('## 4. Latency\n')
    L.append('| Stage (ms) | ground | triage | network | tracking | risk+resample | map update | ray clearing | decision | total | FPS |\n|---|' + '---:|' * 10)
    for k, nm in (('risk', 'TERRA-FOVEA (selective AI)'), ('risk_full_ai', 'network on all points'), ('uniform_5cm', 'uniform 5 cm grid')):
        m = R[k]['mean']
        L.append(f"| {nm} | " + ' | '.join(f"{m[s]:.1f}" for s in ('ms_ground', 'ms_triage', 'ms_net', 'ms_track', 'ms_risk', 'ms_insert', 'ms_rayclear', 'ms_decision', 'ms_total')) + f" | {R[k]['fps']:.1f} |")
    L.append(f"\nSelective AI sends {100 * r['mean']['ai_points'] / r['mean']['points']:.0f}% of the points to the network "
             f"(vs {100 * fa['mean']['ai_points'] / fa['mean']['points']:.0f}%), network time {r['mean']['ms_net']:.1f} ms vs {fa['mean']['ms_net']:.1f} ms, "
             f"mIoU {r['miou']:.3f} vs {fa['miou']:.3f}.\n")
    L.append('![latency](latency.png)\n')
    L.append('## 5. Dynamic objects: ghost removal\n')
    L.append('Occupied map area left where a moving object used to be (and nothing is now).\n')
    L.append('| | mean ghost area | peak ghost area |\n|---|---:|---:|')
    L.append(f"| Ray clearing ON | {r['map']['ghost_m2']:.3f} m2 | {r['ghost_peak_m2']:.3f} m2 |")
    L.append(f"| Ray clearing OFF | {nc['map']['ghost_m2']:.3f} m2 | {nc['ghost_peak_m2']:.3f} m2 |\n")
    if 'hazards' in r:
        L.append('## 6. Hazards a 2D grid cannot represent\n')
        L.append('| Hazard | Truth | Result |\n|---|---|---|')
        for k, v in r['hazards'].items():
            if k.startswith('hole'):
                res = f"flagged at {v['detect_dist']} m range" if v['detect_dist'] is not None else f"NOT flagged (closest approach {v['min_dist']:.1f} m)"
                L.append(f"| {v['kind']} at x={v['x']:.0f} m | {v['depth'] * 100:.0f} cm deep | {res} |")
            elif isinstance(v, dict):
                mc = f"{v['measured_clearance']:.2f} m" if v['measured_clearance'] is not None else 'n/a'
                L.append(f"| {k} | clearance {v['true_clearance']:.2f} m | measured {mc}; obstacle cells wrongly placed on the road under it: {v['max_blocking_cells']} |")
        L.append(f"\nNegative-obstacle false alarms: {r['hazards'].get('neg_false_m2', 0):.3f} m2 of map on average.\n")
    L.append('## 7. Decision layer log\n')
    L.append('| Sweep | Ego x (m) | State | Reason |\n|---:|---:|---|---|')
    for i, st, rs, _, x in r['decisions']:
        L.append(f'| {i} | {x} | {st} | {rs} |')
    if 'robust' in R:
        rb_ = R['robust']
        L.append('\n## 8. Degraded sensor\n')
        L.append(f"30% random point dropout + 3 cm range noise: mIoU {rb_['miou']:.3f} (clean {r['miou']:.3f}), corridor IoU {rb_['map']['ciou']:.3f} (clean {r['map']['ciou']:.3f}), "
                 f"memory {rb_['mean']['map_bytes'] / 1048576:.2f} MB, {rb_['fps']:.1f} FPS.\n")
    open(os.path.join(outdir, 'REPORT.md'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default=os.path.join(ROOT, 'data', 'gauntlet'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'results'))
    ap.add_argument('--frames', type=int, default=0)
    a = ap.parse_args()
    seq = Sequence(a.seq); os.makedirs(a.out, exist_ok=True)
    if seq.labels(0) is None:
        raise SystemExit('this sequence has no .label files; evaluation needs ground truth')
    scene = None
    if seq.simulated and os.path.basename(os.path.normpath(a.seq)) == 'gauntlet':
        import sim
        scene = sim.gauntlet()
    R = {}
    for key, cfg, name in VARIANTS:
        t = time.time(); R[key] = run(seq, cfg, max_frames=a.frames or None, scene=scene)
        v = R[key]
        print(f"{name:38s} {v['mean']['map_bytes'] / 1048576:7.2f} MB  {v['fps']:5.1f} FPS  mIoU {v['miou']:.3f}  corridor IoU {v['map']['ciou']:.3f}  ghost {v['map']['ghost_m2']:.3f} m2  ({time.time() - t:.0f}s)", flush=True)
    R['robust'] = run(seq, 'policy=2', perturb=dict(dropout=0.3, sigma=0.03), max_frames=a.frames or None, scene=scene)
    print(f"degraded sensor: mIoU {R['robust']['miou']:.3f}")
    json.dump(dict(simulated=seq.simulated, sequence=a.seq, results=R), open(os.path.join(a.out, 'metrics.json'), 'w'))
    charts(R, a.out); report(R, a.out, seq)
    print('wrote', os.path.join(a.out, 'REPORT.md'))

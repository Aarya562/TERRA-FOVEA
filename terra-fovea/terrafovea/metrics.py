"""Measurements against simulator ground truth.  Nothing here is used by the pipeline itself."""
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation

from . import classes as C
from .config import Config
from .pipeline import cell_state

BANDS = ((0, 10), (10, 30), (30, 60), (60, 100))


def seg_confusion(label, pred, rng, valid=None):
    """Per-distance-band confusion matrices (8 classes).  pred = -1 counts as a miss."""
    out = []
    for lo, hi in BANDS:
        m = (rng >= lo) & (rng < hi) & (label >= 0)
        if valid is not None:
            m &= valid
        conf = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES + 1), np.int64)
        np.add.at(conf, (label[m].astype(np.int64), np.where(pred[m] >= 0, pred[m], C.NUM_CLASSES).astype(np.int64)), 1)
        out.append(conf)
    return np.stack(out)


def conf_stats(conf):
    """accuracy, mIoU, per-class IoU from a (K, K+1) confusion matrix."""
    sq = conf[:, :C.NUM_CLASSES]
    tp = np.diag(sq).astype(np.float64)
    gt = conf.sum(1); pr = sq.sum(0)
    iou = tp / np.maximum(gt + pr - tp, 1)
    present = gt > 0
    return {"acc": float(tp.sum() / max(conf.sum(), 1)), "miou": float(iou[present].mean()) if present.any() else 0.0,
            "iou": iou, "present": present}


def group_confusion(conf):
    """Collapse an 8-class confusion into the PS's three groups (+ 'unlabelled' column)."""
    g = np.zeros((3, 4), np.int64)
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES + 1):
            g[C.GROUP_OF[i], C.GROUP_OF[j] if j < C.NUM_CLASSES else 3] += conf[i, j]
    return g


def paint(cells, mask, x0, y0, nx, ny, cfg: Config, res=None):
    """Rasterise the selected cells onto a fine raster covering [x0, x0+nx*res) x [y0, y0+ny*res)."""
    res = res or cfg.base_res
    img = np.zeros((nx, ny), bool)
    for L in np.unique(cells["level"][mask]):
        m = mask & (cells["level"] == L)
        size = cfg.level_res(int(L))
        k = int(round(size / res))
        ix0 = np.round((cells["x"][m] - size / 2 - x0) / res).astype(np.int64)
        iy0 = np.round((cells["y"][m] - size / 2 - y0) / res).astype(np.int64)
        off = np.arange(k)
        ix = (ix0[:, None, None] + off[None, :, None]); iy = (iy0[:, None, None] + off[None, None, :])
        ix, iy = np.broadcast_arrays(ix, iy)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        img[ix[ok], iy[ok]] = True
    return img


def map_vs_truth(grid, scene, t, pose, cfg: Config, ahead=(3.0, 60.0), frame_points=None):
    """Compare the map's 'cannot drive here' mask with the truth, on a 5 cm raster of the road ahead.

    inflation_m2   blocked in the map, free in truth, within 0.6 m of a real obstacle (cell too coarse)
    phantom_m2     blocked in the map, free in truth, far from any real obstacle (ghosts / false alarms)
    edge_err_cm    mean distance from a map-blocked pixel to the nearest truly blocked pixel
    recall_obj     fraction of real hazards in view that the map marks at all
    """
    res = cfg.base_res
    x0 = pose[0] + ahead[0]; x1 = pose[0] + ahead[1]
    y0 = -scene.road_half + 0.05; y1 = scene.road_half - 0.05
    nx = int((x1 - x0) / res); ny = int((y1 - y0) / res)
    cells = grid.cells()
    st = cell_state(cells, cfg, ground_z=pose[2] - cfg.sensor_height)
    mb = paint(cells, st["blocked"], x0, y0, nx, ny, cfg)
    xs = x0 + (np.arange(nx) + 0.5) * res; ys = y0 + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    gt, gcls, gdyn, gneg = scene.gt_blocked(X, Y, t, cfg.vehicle_height, cfg.step_height)
    gt_d = binary_dilation(gt, np.ones((3, 3), bool))
    dist = distance_transform_edt(~gt) * res if gt.any() else np.full(gt.shape, 99.0)
    fp = mb & ~gt_d
    infl = fp & (dist <= 0.6)      # 0.6 m > diagonal of the coarsest (40 cm) cell
    phan = fp & (dist > 0.6)
    out = {"blocked_m2": float(mb.sum() * res * res), "truth_m2": float(gt.sum() * res * res),
           "inflation_m2": float(infl.sum() * res * res), "phantom_m2": float(phan.sum() * res * res),
           "edge_err_cm": float(dist[mb].mean() * 100) if mb.any() else 0.0,
           "precision": float((mb & gt_d).sum() / max(mb.sum(), 1)),
           "neg_truth_px": int(gneg.sum()), "neg_found_px": int((paint(cells, st["hole"], x0, y0, nx, ny, cfg) & binary_dilation(gneg, np.ones((5, 5), bool))).sum())}
    return out, (mb, gt, X, Y)

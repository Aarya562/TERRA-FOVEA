"""ctypes binding for the TERRA-FOVEA C++ core + SemanticKITTI-layout sequence loader."""
import ctypes as C
import glob
import json
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NFEAT, CELL_FLOATS, TRACK_FLOATS = 12, 13, 10
CLASS_NAMES = ['unknown', 'drivable', 'terrain', 'structure', 'pole', 'vegetation', 'vehicle', 'person']
STAT_NAMES = ['ms_ground', 'ms_triage', 'ms_net', 'ms_track', 'ms_risk', 'ms_insert', 'ms_rayclear', 'ms_decision',
              'ms_total', 'points', 'ai_points', 'tiles', 'cells_alloc', 'cells_observed', 'map_bytes',
              'level0', 'level1', 'level2', 'level3', 'level4', 'tracks', 'dynamic_tracks', 'cleared', 'splits',
              'merges', 'neg_tiles', 'frame', 'workspace_bytes', 'boosted_tiles', 'relaxed_tiles']
P_AI, P_DYNAMIC, P_NEGATIVE, P_OVERHANG, P_GROUNDLIKE = 1, 2, 4, 8, 16
C_OBSERVED, C_GROUND, C_DYNAMIC, C_NEGATIVE, C_INHERITED = 1, 2, 4, 8, 16


def _lib():
    names = ['libterra_fovea.so', 'libterra_fovea.dylib', 'terra_fovea.dll', 'libterra_fovea.dll']
    for d in ('build', os.path.join('build', 'Release'), os.path.join('build', 'Debug')):
        for nm in names:
            p = os.path.join(ROOT, d, nm)
            if os.path.exists(p):
                return C.CDLL(p)
    sys.exit('engine library not found: run ./build.sh (Linux/macOS) or build.bat (Windows) first')


_L = _lib()
_L.tf_create.restype = C.c_void_p; _L.tf_create.argtypes = [C.c_char_p]
_L.tf_destroy.argtypes = [C.c_void_p]
_L.tf_load_model.argtypes = [C.c_void_p, C.c_char_p]
_L.tf_process.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p, C.c_double]
_L.tf_get_stats.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
_L.tf_point_labels.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_int]
_L.tf_count_cells.argtypes = [C.c_void_p]
_L.tf_export_cells.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
_L.tf_export_tracks.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
_L.tf_decision_json.restype = C.c_char_p; _L.tf_decision_json.argtypes = [C.c_void_p]
_L.tf_export_delta.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int]
_L.tf_features.argtypes = [C.c_void_p] + [C.c_void_p, C.c_int] + [C.c_void_p] * 5
_L.tf_selftest.argtypes = [C.c_int]


def selftest(verbose=True):
    return _L.tf_selftest(1 if verbose else 0)


class Engine:
    def __init__(self, cfg='policy=2', model='auto'):
        self.h = _L.tf_create(cfg.encode())
        self.n = 0
        if model == 'auto':
            model = os.path.join(ROOT, 'models', 'pillar_pointnet.bin')
            if not os.path.exists(model):
                model = None
        self.model = model
        if model and _L.tf_load_model(self.h, model.encode()) != 0:
            raise RuntimeError('cannot load model ' + model)
        self._buf = np.zeros(1 << 22, np.uint8)

    def __del__(self):
        if getattr(self, 'h', None):
            _L.tf_destroy(self.h); self.h = None

    def process(self, pts, pose, t):
        pts = np.ascontiguousarray(pts, np.float32); pose = np.ascontiguousarray(pose, np.float64)
        self.n = len(pts)
        _L.tf_process(self.h, pts.ctypes.data, len(pts), pose.ctypes.data, float(t))

    def stats(self):
        s = np.zeros(len(STAT_NAMES)); _L.tf_get_stats(self.h, s.ctypes.data, len(s))
        return dict(zip(STAT_NAMES, s.tolist()))

    def point_labels(self):
        c = np.zeros(self.n, np.uint8); f = np.zeros(self.n, np.uint8); q = np.zeros(self.n, np.uint8)
        _L.tf_point_labels(self.h, c.ctypes.data, f.ctypes.data, q.ctypes.data, self.n)
        return c, f, q

    def cells(self):
        """columns: x y size ground top clearance class conf occ flags neg level risk"""
        n = _L.tf_count_cells(self.h); out = np.zeros((n, CELL_FLOATS), np.float32)
        k = _L.tf_export_cells(self.h, out.ctypes.data, n)
        return out[:k]

    def tracks(self):
        out = np.zeros((256, TRACK_FLOATS), np.float32)
        return out[:_L.tf_export_tracks(self.h, out.ctypes.data, 256)]

    def decision(self):
        return json.loads(_L.tf_decision_json(self.h).decode())

    def delta(self, full=False):
        k = _L.tf_export_delta(self.h, self._buf.ctypes.data, len(self._buf), 1 if full else 0)
        if k > len(self._buf):
            self._buf = np.zeros(k * 2, np.uint8)
            k = _L.tf_export_delta(self.h, self._buf.ctypes.data, len(self._buf), 1)
        return self._buf[:k].tobytes()

    def features(self, pts, pose):
        pts = np.ascontiguousarray(pts, np.float32); pose = np.ascontiguousarray(pose, np.float64); n = len(pts)
        f = np.zeros((n, NFEAT), np.float32); g1 = np.zeros(n, np.int32); g2 = np.zeros(n, np.int32); tri = np.zeros(n, np.uint8)
        _L.tf_features(self.h, pts.ctypes.data, n, pose.ctypes.data, f.ctypes.data, g1.ctypes.data, g2.ctypes.data, tri.ctypes.data)
        return f, g1, g2, tri


# ----------------------------------------------------------------------------- data
# SemanticKITTI id -> TERRA-FOVEA class  (201 is the simulator's "negative obstacle" surface)
_MAP = {0: 0, 1: 0, 10: 6, 11: 6, 13: 6, 15: 6, 16: 6, 18: 6, 20: 6, 30: 7, 31: 7, 32: 7, 40: 1, 44: 1, 48: 2, 49: 2,
        50: 3, 51: 3, 52: 3, 60: 1, 70: 5, 71: 5, 72: 2, 80: 4, 81: 4, 99: 3, 201: 2,
        252: 6, 253: 7, 254: 7, 255: 7, 256: 6, 257: 6, 258: 6, 259: 6}
LUT = np.zeros(65536, np.uint8)
for _k, _v in _MAP.items():
    LUT[_k] = _v
MOVING = np.zeros(65536, bool); MOVING[252:260] = True


class Sequence:
    """<dir>/velodyne/*.bin, optional labels/*.label, poses.txt, calib.txt, times.txt (SemanticKITTI layout)."""

    def __init__(self, d):
        self.dir = d
        self.bins = sorted(glob.glob(os.path.join(d, 'velodyne', '*.bin')))
        if not self.bins:
            sys.exit(f'no scans in {d}/velodyne (run python/sim.py, or point to a SemanticKITTI sequence)')
        Tr = np.eye(4)
        cp = os.path.join(d, 'calib.txt')
        if os.path.exists(cp):
            for line in open(cp):
                if line.startswith('Tr:'):
                    Tr[:3, :] = np.array(line.split()[1:13], float).reshape(3, 4)
        pp = os.path.join(d, 'poses.txt')
        if os.path.exists(pp):
            raw = np.loadtxt(pp).reshape(-1, 3, 4)
            Tri = np.linalg.inv(Tr); self.poses = []
            for r in raw:
                P = np.eye(4); P[:3, :] = r
                self.poses.append(Tri @ P @ Tr)
        else:                                                     # no odometry: sensor frame = world frame
            self.poses = [np.eye(4)] * len(self.bins)
        tp = os.path.join(d, 'times.txt')
        self.times = np.loadtxt(tp).tolist() if os.path.exists(tp) else [0.1 * i for i in range(len(self.bins))]
        self.simulated = os.path.exists(os.path.join(d, 'SIMULATED.txt'))

    def __len__(self):
        return min(len(self.bins), len(self.poses))

    def scan(self, i):
        return np.fromfile(self.bins[i], np.float32).reshape(-1, 4)

    def labels(self, i):
        p = self.bins[i].replace('velodyne', 'labels').replace('.bin', '.label')
        if not os.path.exists(p):
            return None
        return (np.fromfile(p, np.uint32) & 0xFFFF).astype(np.int64)

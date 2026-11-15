"""SemanticKITTI reader (also reads RELLIS-3D, which ships in the same layout).

Expected folder:
    <sequence>/velodyne/000000.bin    float32 [x, y, z, intensity] per point, sensor frame
    <sequence>/labels/000000.label    uint32, low 16 bits = semantic id          (optional)
    <sequence>/poses.txt              one 3x4 camera pose per line               (optional)
    <sequence>/calib.txt              line 'Tr: ...' = velodyne -> camera 3x4    (optional)

Each item is a sweep dict in the format Pipeline.process() takes.  The full 4x4 sensor->world
transform is passed as 'T' so roll and pitch are respected when points are placed in the map.
"""
import os
import numpy as np

from . import classes as C

_LUT = np.full(260, -1, np.int8)
for _k, _v in C.SEMKITTI_MAP.items():
    _LUT[_k] = _v
_MOVING = np.zeros(260, bool)
for _k in C.SEMKITTI_MOVING:
    _MOVING[_k] = True


def read_calib(path):
    with open(path) as f:
        for line in f:
            if line.startswith("Tr"):
                v = np.array(line.split(":")[1].split(), dtype=np.float64).reshape(3, 4)
                T = np.eye(4); T[:3] = v
                return T
    return np.eye(4)


def read_poses(path, Tr):
    """Camera poses -> sensor poses, expressed relative to the first sweep."""
    P = np.loadtxt(path, dtype=np.float64).reshape(-1, 3, 4)
    Tr_inv = np.linalg.inv(Tr)
    out = []
    for p in P:
        T = np.eye(4); T[:3] = p
        out.append(Tr_inv @ T @ Tr)
    return out


class KittiSequence:
    def __init__(self, folder, hz=10.0, sensor_height=1.73):
        self.folder = folder
        self.hz = hz
        self.sensor_height = sensor_height
        vd = os.path.join(folder, "velodyne")
        self.files = sorted(f for f in os.listdir(vd) if f.endswith(".bin"))
        cal = os.path.join(folder, "calib.txt"); pos = os.path.join(folder, "poses.txt")
        Tr = read_calib(cal) if os.path.exists(cal) else np.eye(4)
        self.poses = read_poses(pos, Tr) if os.path.exists(pos) else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        raw = np.fromfile(os.path.join(self.folder, "velodyne", name), dtype=np.float32).reshape(-1, 4)
        T = self.poses[i] if self.poses is not None and i < len(self.poses) else np.eye(4)
        T = T.copy(); T[2, 3] += self.sensor_height          # world z = 0 at the road under the first sweep
        yaw = float(np.arctan2(T[1, 0], T[0, 0]))
        sweep = {"points": np.ascontiguousarray(raw[:, :3]), "intensity": np.ascontiguousarray(raw[:, 3]),
                 "pose": np.array([T[0, 3], T[1, 3], T[2, 3], yaw]), "T": T, "t": i / self.hz}
        lab = os.path.join(self.folder, "labels", name.replace(".bin", ".label"))
        if os.path.exists(lab):
            sem = np.fromfile(lab, dtype=np.uint32) & 0xFFFF
            sem = np.minimum(sem, 259)
            sweep["label"] = _LUT[sem]
            sweep["moving"] = _MOVING[sem]
        return sweep


def write_sequence(folder, sweeps, sensor_height=1.9):
    """Write sweeps (e.g. from the simulator) in SemanticKITTI layout.  Used by the tests."""
    os.makedirs(os.path.join(folder, "velodyne"), exist_ok=True)
    os.makedirs(os.path.join(folder, "labels"), exist_ok=True)
    inv = {v: k for k, v in [(40, C.ROAD), (72, C.TERRAIN), (50, C.BUILDING), (80, C.POLE), (70, C.VEGETATION),
                              (10, C.VEHICLE), (30, C.PEDESTRIAN), (99, C.OTHER)]}
    lines = []
    for i, s in enumerate(sweeps):
        np.concatenate([s["points"], s["intensity"][:, None]], 1).astype(np.float32).tofile(
            os.path.join(folder, "velodyne", f"{i:06d}.bin"))
        np.array([inv[int(c)] for c in s["label"]], dtype=np.uint32).tofile(os.path.join(folder, "labels", f"{i:06d}.label"))
        x, y, z, yaw = s["pose"]; c, sn = np.cos(yaw), np.sin(yaw)
        T = np.array([[c, -sn, 0, x], [sn, c, 0, y], [0, 0, 1, z - sensor_height]])
        lines.append(" ".join(f"{v:.9e}" for v in T.reshape(-1)))
    open(os.path.join(folder, "poses.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(folder, "calib.txt"), "w").write("Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n")

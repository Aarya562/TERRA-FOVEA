"""Semantic classes used across the whole project."""
import numpy as np

ROAD, TERRAIN, BUILDING, POLE, VEGETATION, VEHICLE, PEDESTRIAN, OTHER = range(8)
NUM_CLASSES = 8
NAMES = ["road", "terrain", "building", "pole", "vegetation", "vehicle", "pedestrian", "other"]

# The three groups the problem statement asks for.
G_TERRAIN, G_STATIC, G_DYNAMIC = 0, 1, 2
GROUP_NAMES = ["terrain", "static obstacle", "dynamic object"]
GROUP_OF = np.array([0, 0, 1, 1, 1, 2, 2, 1], dtype=np.int64)

GROUND_CLASSES = (ROAD, TERRAIN)
DYNAMIC_CLASSES = (VEHICLE, PEDESTRIAN)

# RGB colours shared by the dashboard and the report plots.
COLORS = np.array([
    [70, 78, 92],      # road        - slate
    [96, 140, 82],     # terrain     - green
    [214, 138, 62],    # building    - orange
    [240, 214, 80],    # pole        - yellow
    [52, 168, 108],    # vegetation  - bright green
    [66, 148, 255],    # vehicle     - blue
    [255, 76, 110],    # pedestrian  - red/pink
    [170, 120, 220],   # other       - purple
], dtype=np.uint8)

# SemanticKITTI raw label id -> our class (-1 = ignore).
SEMKITTI_MAP = {
    0: -1, 1: -1,
    10: VEHICLE, 11: VEHICLE, 13: VEHICLE, 15: VEHICLE, 16: VEHICLE, 18: VEHICLE, 20: VEHICLE,
    30: PEDESTRIAN, 31: PEDESTRIAN, 32: PEDESTRIAN,
    40: ROAD, 44: ROAD, 48: TERRAIN, 49: TERRAIN, 60: ROAD,
    50: BUILDING, 51: BUILDING, 52: OTHER,
    70: VEGETATION, 71: POLE, 72: TERRAIN,
    80: POLE, 81: POLE, 99: OTHER,
    252: VEHICLE, 253: PEDESTRIAN, 254: PEDESTRIAN, 255: PEDESTRIAN,
    256: VEHICLE, 257: VEHICLE, 258: VEHICLE, 259: VEHICLE,
}
SEMKITTI_MOVING = {252, 253, 254, 255, 256, 257, 258, 259}

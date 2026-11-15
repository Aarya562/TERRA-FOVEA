# How TERRA-FOVEA works

Read this first. Each section says what the stage does, why, and where the code is
(`C++` = `core/terra_fovea.cpp`, `ref` = the `terrafovea/` NumPy reference).

## The problem in one paragraph

A 64-beam LiDAR produces about 100,000 points ten times a second. Keeping all of them in a 5 cm 3D grid
costs gigabytes. A flat 2D grid is cheap but cannot tell a kerb from a wall, a pit from flat road, or a
bridge from a barrier. A 2.5D map stores a few heights per ground cell instead of a full column of voxels.
Variable resolution means small cells near the vehicle and large cells far away, like the eye's fovea.

## Stage 1. Ground model and triage (C++ sections 1-2, ref `ground.py`)

Every point is compared with a robust local ground surface and sorted into four kinds:

| Kind | Meaning |
|---|---|
| ground | within a few centimetres of the surface |
| obstacle | above the surface and below 2.5 m: it blocks the vehicle |
| overhead | 2.5 m or more above the surface: the vehicle may pass underneath |
| below | under the surface: the signature of a pit or trench |

The reference fits a plane per 1 m cell with re-weighted least squares. Vehicles, kerbs and pits are
a minority in the fitting window, so they stop influencing the plane. Two details that came from bugs
found during testing:

- Far away, ground returns are metres apart, and a sign gantry can be the only thing in a cell. A radial
  profile grown outward from the vehicle stops such a cell being mistaken for ground.
- Where a road meets a raised footpath, the lower surface is the ground. The fit tolerates cells below the
  plane more than cells above it.

Smooth, flat ground is called drivable by geometry alone. Rough or steep ground is called terrain.

## Stage 2. Selective neural segmentation (C++ section 3, ref `model.py`)

The network is a PointNet-style model with pillar pooling: a shared MLP per point, max-pooling inside
vertical pillars at several scales for context, then an MLP that outputs 8 classes. Pillar grouping is
O(N), which is why it runs on a CPU where PointNet++ ball queries would not.

It is only run where geometry cannot decide: obstacle points and ground that is neither clearly smooth nor
clearly rough. In the C++ report that is 59% of the points for almost the same mIoU.

The reference contains the full forward pass, the hand-derived backward pass and Adam in NumPy, checked
against numerical gradients in `tests/test_model_gradients.py`.

## Stage 3. Risk plan and exact regrid (C++ section 5, ref `policy.py`, `grid.py`)

The world is cut into tiles. Each tile holds cells at one level: 5, 10, 20, 40 or 80 cm.

**Distance bands** come from sensor physics. A 0.2 degree LiDAR puts neighbouring returns r x 0.0035 m
apart, so beyond about 14 m one sweep cannot fill a 5 cm cell. Finer cells there store nothing new.

**Risk overrides distance.** A tile is refined when it is in the path the vehicle is about to drive, holds
an obstacle or suspected pit in that path, holds a moving object, or lies where a moving object is predicted
to be in the next seconds.

**Why 40 cm and not 50 cm.** 5, 10, 20, 40 nest exactly: each cell is four cells of the level below. 50 cm does
not nest with 20 cm, and that mismatch is exactly the alignment error the problem statement warns about.
40 cm is finer than the 50 cm the statement gives as an example.

**No data loss.** Cells inside a tile are stored in Z-order (Morton order), so the four children of a cell
are four consecutive entries. Coarsening is a reshape plus a reduction; refining is a repeat. The rules:

| Field | When four cells merge |
|---|---|
| obstacle top | maximum: nothing that blocks can vanish |
| overhead underside | minimum: clearance never looks better than it is |
| occupancy | never decreases |
| class evidence | per-class maximum: one pedestrian cell is not out-voted by three road cells |
| pit confidence | maximum |

Refining is immediate. Coarsening waits a few frames so tiles do not flicker. A hard cell budget coarsens
the lowest-risk, farthest tiles first, so memory is bounded in any scene. These properties are unit tests
in `tests/test_grid.py` and in `terra_cli selftest`.

## Stage 4. Multi-surface fusion (C++ section 6, ref `grid.py`)

Each cell keeps ground height, the top of the lowest blocking obstacle, the underside of anything overhead,
occupancy, class evidence, dynamic flag and pit confidence: 16 bytes in C++, 27 in the reference.
That is what lets the map say a sign gantry at 5 m is passable and a boom at 1.7 m is not.

## Stage 5. Tracking (C++ section 4, ref `tracking.py`)

Points classed vehicle or pedestrian are clustered and tracked with a constant-velocity Kalman filter.
A parked car that the ego drives past looks as if it moves, because more of it comes into view. The
reference only calls a track moving when both opposite edges of its footprint move the same way.

## Stage 6. Ghost clearing (C++ section 7, ref `visibility.py`)

The sweep is rendered as a small range image. A map cell that the laser has just seen through, at the
height of its stored obstacle, is no longer occupied. Cells hidden behind something are left alone.
Space a tracked vehicle has just driven through is released as well, since nothing solid can be standing there.

## Stage 7. Decision (C++ section 8)

The driving corridor is checked for blocking cells, pits, low clearance and predicted crossings, and the
engine reports CLEAR, CAUTION or STOP with a reason and a distance. The reference has no decision stage.

## How this maps to the problem statement

| The statement asks for | Where |
|---|---|
| Terrain analysis, drivable or not | Stages 1 and 2, the Drivable view in the dashboard |
| Static and dynamic object classification | Stages 2 and 5 |
| Non-uniform grid without alignment error or data loss | Stage 3, proven by tests |
| 5 cm within 10 m, coarser to 100 m | Stage 3 bands |
| Real-time visualisation with memory comparison | `dashboard/`, the memory panel |
| Latency and accuracy across distance | `results/REPORT.md`, `results/reference/RESULTS.md` |

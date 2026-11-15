# Questions the jury is likely to ask

Answer from what the code does. If you do not know, say so and offer to show the file.

**Is this running on real LiDAR data?**
Say what is true on the day. As shipped: no, all numbers are from a simulator with exact ground truth,
and the loader reads SemanticKITTI layout. After you follow `docs/REAL_DATA.md` you can answer with real numbers.

**Why 2.5D and not 3D voxels?**
A ground vehicle needs, per patch of ground: where the ground is, what blocks it, and what hangs over it.
Three heights per cell answer that. A dense 5 cm voxel grid of the same area is about 1.2 GB; our map is under 10 MB.

**A 2.5D map loses overhangs. How do you handle a bridge or a tree?**
Each cell stores the underside of anything above 2.5 m separately from the obstacle top. A gantry leaves the
cell drivable with a known clearance; a low boom makes it blocked. See section 6 of `results/REPORT.md`.

**How do you avoid alignment errors where resolution changes?**
Every level is exactly twice the cell size of the one below and all levels share one origin, so a coarse cell
is exactly four finer cells. There is no interpolation anywhere. `terra_cli selftest` and `tests/test_grid.py`
check that every point lands in exactly one cell and that refine-then-coarsen returns the identical data.

**The statement says 50 cm. Why is your coarse cell 40 cm?**
50 does not nest with 20. Using it would create the alignment error the statement warns about. 40 cm is finer
than required. The base size is one config value if a different pyramid is wanted.

**What happens to a pedestrian when cells are merged?**
Class evidence is merged by per-class maximum, and the obstacle top by maximum. One pedestrian cell among
three road cells still reports a pedestrian-height obstacle. Also, the risk policy keeps tiles with moving
objects fine in the first place.

**Other teams also adapt resolution. What is different?**
Three things we can show running: resolution follows risk (path, hazards, predicted motion) and not only
distance; the map has ground, obstacle and overhead surfaces plus pit confidence; and every claim has a
baseline and a test. Ask them for their memory figure against a sparse map, not a dense one.

**Your memory saving is only 10x. Another team claims 250x.**
Against a dense 5 cm voxel grid ours is about 156x. We also report against the honest baseline, the same
sparse engine at uniform 5 cm, which is about 10x. A dense grid is a baseline nobody would deploy.

**Why not SPVCNN or Cylinder3D?**
They need a GPU and a sparse-convolution library to reach real time. Our network runs in the C++ core on
CPU threads in about 30 ms. The interface is per-point class scores, so a heavier network can replace it
on hardware that has a GPU. We did not benchmark SPVCNN ourselves; do not claim that we did.

**What is "selective AI"?**
Geometry first labels what it can: smooth flat ground, and the split into ground, obstacle, overhead, below.
The network runs only on the rest. That cut network input to 59% of points with mIoU 0.947 against 0.956.

**How do you detect potholes with no training data for them?**
Geometry. The ground plane ignores the pit, so returns from inside it come out below the plane. Evidence
accumulates over sweeps into a confidence. Limit: at long range the beam grazes the surface and cannot see
into the pit, so detection happens at roughly 6 to 10 m.

**How are ghost trails removed? Is that Bresenham ray tracing?**
Same physics, cheaper. The sweep is rendered as a range image. For each occupied cell not hit this sweep we
look up the beam in that direction: if it travelled clearly beyond the cell at the obstacle's height, the cell
is cleared. That is one lookup per occupied cell and not a walk along 100,000 rays.

**Is it real time?**
The C++ engine takes about 56 ms per sweep on two CPU cores, 17.8 FPS, against a 10 Hz sensor. The Python
reference is about 6 FPS and exists for readability.

**Where does the pose come from?**
From the dataset (or the simulator). Localisation is outside this project. On a vehicle it would come from
LiDAR odometry or GNSS/INS.

**Does it run on a Jetson? ROS2? TensorRT?**
Not tested. The core is plain C++17 with a C API and no dependencies, so wrapping it in a ROS2 node is
straightforward, but we have not done it. Do not claim otherwise.

**What fails?**
Steep broken terrain breaks the planar ground assumption. Far, sparse objects are classified worse: moving-object
recall drops beyond 60 m. Glass and rain are not modelled. The network has only seen simulated data.

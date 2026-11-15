# Running on real LiDAR (SemanticKITTI)

None of this has been executed in the environment the project was built in: it had no dataset, no GPU and
no PyTorch. Treat it as a procedure to follow and debug, and budget a day for it.

## 1. Get data
SemanticKITTI = KITTI odometry velodyne scans + SemanticKITTI labels + poses. On Kaggle, search for a
"SemanticKITTI" dataset and attach it to a notebook; you need the folder that contains `sequences/00 ... 10`.
Each sequence has `velodyne/*.bin`, `labels/*.label`, `poses.txt`, `calib.txt`.
RELLIS-3D (off-road) ships in the same layout if you want terrain closer to the defence use case.

## 2. First look, no training
```
python scripts/run_dashboard.py --engine python --seq <path>/sequences/08 --sensor-height 1.73
```
The map geometry (ground, obstacles, overhead, variable resolution, memory) does not depend on the network
and should already look right. Classes will be poor because the network has only seen the simulator.
Check: is the road flat and marked drivable, are walls and cars raised, does memory stay in the low tens of MB.

## 3. Train
Two networks exist, one per engine.

- Reference engine: `scripts/train_torch.py --data <path>/sequences --epochs 6` on a Kaggle or Colab GPU.
  It writes `weights/pillarpointnet_kitti.npz`; pass it with `--weights`. This script has never been run.
  Likely first-run fixes: tensor dtype or device mismatches, and `scatter_reduce` needs PyTorch 1.12 or newer.
- C++ engine: `python python/train.py --data <folder with train_* and val_* sequences>` trains in NumPy using
  features computed by the C++ core and writes `models/pillar_pointnet.bin`. Name or link your KITTI sequences
  as `train_00`, `train_01`, ..., `val_00`, and use `--every 5` to subsample.

## 4. Things that differ from the simulator
- Sensor height 1.73 m, vertical field of view +2 to -24.8 degrees.
- Intensity statistics differ; that is why retraining is needed.
- Poses include roll and pitch. Both engines take the full 4x4 pose.
- Labels: the mapping from SemanticKITTI ids to our 8 classes is in `terrafovea/classes.py` and `python/engine.py`.
  The reference has a class "other"; the C++ engine folds it into "structure".

## 5. Evaluate
`python python/evaluate.py --seq <path>/sequences/08` is the harness to use. It was written for the simulator, whose
files carry exact geometry; the map-footprint, pit and clearance sections depend on that and will need adapting
or skipping on real data. Segmentation, memory and latency only need the label files.
Report real and simulated numbers separately and label each.

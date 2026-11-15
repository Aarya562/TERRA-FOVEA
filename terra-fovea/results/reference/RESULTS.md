# Measured results

Scene `urban_demo`, 140 sweeps at 10 Hz, about 96892 returns per sweep. Every row saw the same sweeps. Timing is from the machine that ran the benchmark (CPU only).

| Variant | Memory | FPS | Over-inflation | Phantom area | Edge error | Point accuracy | mIoU |
|---|---|---|---|---|---|---|---|
| Uniform 5 cm | 213 MB | 4.4 | 0.15 m² | 0.09 m² | 8.4 cm | 93.9% | 77.8% |
| Uniform 40 cm | 3.4 MB | 7.3 | 5.17 m² | 6.91 m² | 145.5 cm | 93.9% | 77.8% |
| Distance-only adaptive | 10.4 MB | 6.6 | 2.16 m² | 0.88 m² | 39.4 cm | 93.9% | 77.8% |
| TERRA-FOVEA (risk-adaptive) | 12.5 MB | 6.4 | 0.82 m² | 0.63 m² | 49.1 cm | 93.9% | 77.8% |
| A1_no_ray_clearing | 13.5 MB | 7.0 | 1.65 m² | 24.03 m² | 318.9 cm | 93.9% | 77.8% |
| A2_ai_on_all_points | 12.5 MB | 5.1 | 0.84 m² | 0.63 m² | 49.1 cm | 93.3% | 76.8% |
| A3_geometry_only | 16.7 MB | 9.9 | 0.88 m² | 0.59 m² | 44.9 cm | 36.3% | 11.3% |

## Memory baselines

| Representation | Memory | vs TERRA-FOVEA |
|---|---|---|
| Dense 3D voxel grid, 5 cm (computed) | 1.60 GB | 128.4x |
| Dense 2.5D grid, 5 cm (computed) | 432 MB | 34.7x |
| Uniform 5 cm 2.5D, sparse tiles (measured) | 213 MB | 17.1x |
| Sparse 3D voxel map, 5 cm (measured) | 27.4 MB | 2.2x |
| TERRA-FOVEA (measured) | 12.5 MB | 1.0x |
| Distance-only adaptive (measured) | 10.4 MB | 0.8x |
| Uniform 40 cm 2.5D (measured) | 3.4 MB | 0.3x |

## TERRA-FOVEA by distance

| Band | Points | Accuracy | mIoU | Terrain | Static | Dynamic |
|---|---|---|---|---|---|---|
| 0-10 m | 7,871,384 | 95.3% | 80.6% | 99.9% | 93.0% | 97.5% |
| 10-30 m | 3,416,737 | 90.1% | 71.7% | 99.8% | 95.9% | 95.4% |
| 30-60 m | 654,617 | 95.7% | 71.3% | 99.6% | 96.8% | 74.6% |
| 60-100 m | 172,811 | 95.3% | 59.4% | 99.7% | 94.5% | 41.4% |

Per-class IoU: road 90%, terrain 82%, building 95%, pole 79%, vegetation 92%, vehicle 92%, pedestrian 77%, other 15%

## Moving objects

Recall 88%, precision 82% over 356 object-sweeps within 50 m; mean speed error 0.33 m/s.

## Latency (TERRA-FOVEA)

| Stage | Mean | 95th percentile |
|---|---|---|
| Geometry triage | 35.8 ms | 45.8 ms |
| Neural segmentation | 67.5 ms | 82.3 ms |
| Risk plan + regrid | 6.5 ms | 9.9 ms |
| 2.5D fusion | 18.3 ms | 24.1 ms |
| Tracking | 8.6 ms | 11.5 ms |
| Ghost clearing | 19.5 ms | 25.8 ms |
| Total | 156.2 ms | 190.7 ms |

## Robustness: 30% extra dropout and +3 cm range noise

Accuracy 94.5% (clean 93.9%), mIoU 78.5%, over-inflation 1.04 m², phantom area 0.81 m², moving-object recall 84%.

## What the columns mean

- Over-inflation: road area the map marks blocked that is actually free, within 0.6 m of a real obstacle. Coarse cells cause it.
- Phantom area: road area marked blocked with no real obstacle within 0.6 m. Ghost trails and false alarms cause it.
- Edge error: mean distance from a blocked map pixel to the nearest truly blocked pixel.
- All three are measured on a 5 cm raster of the road from 3 m to 60 m ahead of the vehicle, against the simulator's exact geometry.

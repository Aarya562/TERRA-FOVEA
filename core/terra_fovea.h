/* TERRA-FOVEA core :: C API
 * Risk-adaptive, variable-resolution 2.5D LiDAR mapping engine.
 * Dependency-free C++17 behind a plain C interface (usable from C, C++, Python ctypes, ROS2 nodes).
 */
#ifndef TERRA_FOVEA_H
#define TERRA_FOVEA_H
#include <stdint.h>

#if defined(_WIN32)
#  define TF_API __declspec(dllexport)
#else
#  define TF_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TFEngine TFEngine;

/* semantic classes */
enum { TF_UNKNOWN = 0, TF_DRIVABLE = 1, TF_TERRAIN = 2, TF_STRUCTURE = 3, TF_POLE = 4,
       TF_VEGETATION = 5, TF_VEHICLE = 6, TF_PERSON = 7, TF_NCLS = 8 };

/* per-point flags returned by tf_point_labels */
enum { TF_P_AI = 1, TF_P_DYNAMIC = 2, TF_P_NEGATIVE = 4, TF_P_OVERHANG = 8, TF_P_GROUNDLIKE = 16 };

/* per-cell flags */
enum { TF_C_OBSERVED = 1, TF_C_GROUND = 2, TF_C_DYNAMIC = 4, TF_C_NEGATIVE = 8, TF_C_INHERITED = 16 };

/* indices into the stats array filled by tf_get_stats */
enum {
  TF_S_MS_GROUND = 0, TF_S_MS_TRIAGE, TF_S_MS_NET, TF_S_MS_TRACK, TF_S_MS_RISK, TF_S_MS_INSERT,
  TF_S_MS_RAYCLEAR, TF_S_MS_DECISION, TF_S_MS_TOTAL,
  TF_S_POINTS, TF_S_AI_POINTS, TF_S_TILES, TF_S_CELLS_ALLOC, TF_S_CELLS_OBSERVED, TF_S_MAP_BYTES,
  TF_S_LEVEL0, TF_S_LEVEL1, TF_S_LEVEL2, TF_S_LEVEL3, TF_S_LEVEL4,
  TF_S_TRACKS, TF_S_DYNAMIC_TRACKS, TF_S_CLEARED, TF_S_SPLITS, TF_S_MERGES, TF_S_NEG_CELLS,
  TF_S_FRAME, TF_S_WORKSPACE_BYTES, TF_S_BOOSTED_TILES, TF_S_RELAXED_TILES, TF_S_COUNT
};

#define TF_NFEAT 12          /* per-point input features of the network */
#define TF_CELL_FLOATS 13    /* floats per cell written by tf_export_cells */
#define TF_TRACK_FLOATS 10   /* floats per track written by tf_export_tracks */

/* cfg: "key=value;key=value". Keys: policy (0 uniform,1 distance,2 risk), uniform_level, r0..r3,
 * max_range, selective_ai, ray_clear, veh_height, veh_width, rough_thr, threads */
TF_API TFEngine*   tf_create(const char* cfg);
TF_API void        tf_destroy(TFEngine*);
TF_API int         tf_load_model(TFEngine*, const char* path);

/* points: n x 4 float (x y z intensity) in the sensor frame; pose: row-major 4x4 sensor->world */
TF_API int         tf_process(TFEngine*, const float* points, int n, const double* pose, double t);

TF_API void        tf_get_stats(const TFEngine*, double* out, int n);
TF_API int         tf_point_labels(const TFEngine*, uint8_t* cls, uint8_t* flags, uint8_t* conf, int n);
TF_API int         tf_count_cells(const TFEngine*);
/* per cell: x y size ground top clearance class conf occ flags neg level risk  (clearance < 0 = none) */
TF_API int         tf_export_cells(const TFEngine*, float* out, int cap_cells);
/* per track: id x y vx vy extent class dynamic age path_relevant */
TF_API int         tf_export_tracks(const TFEngine*, float* out, int cap);
TF_API const char* tf_decision_json(const TFEngine*);
/* delta stream for the dashboard; full!=0 forces a key frame. Returns bytes written (or needed if > cap). */
TF_API int         tf_export_delta(TFEngine*, uint8_t* out, int cap, int full);

/* feature extraction only (training): feats n x TF_NFEAT, g1/g2 pooling group ids, triage 0=skip 1=AI 2=below */
TF_API int         tf_features(TFEngine*, const float* points, int n, const double* pose,
                               float* feats, int32_t* g1, int32_t* g2, uint8_t* triage);

TF_API int         tf_selftest(int verbose);     /* returns number of failed checks */
TF_API const char* tf_version(void);

#ifdef __cplusplus
}
#endif
#endif

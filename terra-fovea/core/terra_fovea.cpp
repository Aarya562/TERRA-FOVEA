// TERRA-FOVEA core engine  (C++17, no dependencies)
//
// Pipeline per LiDAR sweep
//   1 ground model      robust tile-wise ground surface (5x5 median of tile minima + refinement)
//   2 geometric triage  height / roughness / column structure per 20 cm pillar; decides which
//                       points are "obviously smooth ground" and which need the network
//   3 selective AI      Pillar-PointNet forward pass on the uncertain points only
//   4 objects           class-aware clustering + alpha-beta tracking -> dynamic objects
//   5 risk -> resolution per-tile target level from distance + path + motion + hazard + uncertainty
//   6 map update        multi-surface 2.5D cells (ground, obstacle top, overhead clearance)
//   7 ray clearing      rays that pass through a stored obstacle column remove it (no ghosts)
//   8 decision          corridor check -> CLEAR / CAUTION / STOP with a stated reason
//
// Map structure
//   The world is tiled by 0.8 m tiles. Each tile is stored at ONE level L (cell = 5cm * 2^L,
//   L = 0..4) as a dense array in Z-order (Morton order). Because every level divides the tile
//   exactly, tiles of different resolution always meet on a shared edge: no overlap, no gap.
//   Z-order makes coarsening a linear pass: parent i = children 4i..4i+3.
#include "terra_fovea.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
static inline int tf_max_threads() { return omp_get_max_threads(); }
static inline int tf_thread_id() { return omp_get_thread_num(); }
#else
static inline int tf_max_threads() { return 1; }
static inline int tf_thread_id() { return 0; }
#endif

namespace {

const double TILE = 0.8;          // tile edge (m)
const double CELL0 = 0.05;        // finest cell (m)
const int LMAX = 4;               // coarsest level: one 0.8 m cell per tile
const int DIRN = 264;             // rolling tile directory (torus), 211 m span
const int RI_ROWS = 160, RI_COLS = 1024; const float RI_EL0 = 0.70f, RI_DEL = 0.0074f;   // range image: -40..+28 deg
const int WT = 272;               // frame-local window in tiles (+-108.8 m)
const int WP = WT * 4;            // frame-local window in 20 cm pillars
const float PILLAR = 0.2f;
const int OCC_HIT = 12, OCC_MAX = 120, OCC_CLEAR = 10;

inline int cps(int level) { return 16 >> level; }            // cells per tile side
inline int ncell(int level) { return 256 >> (2 * level); }   // cells per tile
inline double csize(int level) { return CELL0 * (1 << level); }

const uint8_t MORT[16] = {0, 1, 4, 5, 16, 17, 20, 21, 64, 65, 68, 69, 80, 81, 84, 85};
inline int morton(int cx, int cy) { return MORT[cx] | (MORT[cy] << 1); }
inline void demorton(int m, int& cx, int& cy) {
  cx = cy = 0;
  for (int b = 0; b < 4; b++) { cx |= ((m >> (2 * b)) & 1) << b; cy |= ((m >> (2 * b + 1)) & 1) << b; }
}
inline int popcnt(uint32_t v) { int c = 0; while (v) { v &= v - 1; c++; } return c; }
inline int ctz32(uint32_t v) { int c = 0; while (!(v & 1u)) { v >>= 1; c++; } return c; }
inline int floordiv(double v, double s) { return (int)std::floor(v / s); }
inline int wrap(int v) { int r = v % DIRN; return r < 0 ? r + DIRN : r; }

// ------------------------------------------------------------------------------- cell
#pragma pack(push, 1)
struct Cell {            // 16 bytes
  int16_t g;             // ground height (cm, world z)
  uint8_t top;           // obstacle top above ground, 2 cm units (0 = no obstacle)
  uint8_t clear;         // lowest overhead return above ground, 2 cm units (255 = open sky)
  uint32_t hist;         // semantic evidence, 8 classes x 4 bit saturating counters
  uint8_t occ;           // occupancy evidence q = -ln(1-p) * 32  (additive under merge)
  uint8_t flags;         // TF_C_*
  uint8_t neg;           // negative-obstacle confidence 0..255
  uint8_t grange;        // ground height spread inside the cell (cm)
  uint8_t age;           // number of sweeps this cell has been hit by an obstacle return (saturating)
  uint8_t spare;
  uint8_t gcnt;          // ground observation count (saturating)
  uint8_t ostamp;        // low 8 bits of the frame of the last obstacle hit
};
#pragma pack(pop)

inline int hget(uint32_t h, int c) { return (h >> (4 * c)) & 15; }
inline void hadd(uint32_t& h, int c, int w) {
  int v = hget(h, c) + w;
  if (v > 15) {                               // halve everything: keeps ratios, frees headroom
    uint32_t o = 0;
    for (int k = 0; k < 8; k++) o |= (uint32_t)(hget(h, k) >> 1) << (4 * k);
    h = o; v = std::min(15, hget(h, c) + w);
  }
  h = (h & ~(15u << (4 * c))) | ((uint32_t)v << (4 * c));
}
inline bool has_obstacle(const Cell& c) { return c.occ > 0 && c.top > 0; }
inline int cell_class(const Cell& c, int* conf = nullptr) {
  int lo = has_obstacle(c) ? 3 : 1, hi = has_obstacle(c) ? 7 : 2, best = 0, bv = 0, sum = 0;
  for (int k = 1; k < 8; k++) sum += hget(c.hist, k);
  for (int k = lo; k <= hi; k++) if (hget(c.hist, k) > bv) { bv = hget(c.hist, k); best = k; }
  if (!best) for (int k = 1; k < 8; k++) if (hget(c.hist, k) > bv) { bv = hget(c.hist, k); best = k; }
  if (conf) *conf = sum ? (255 * bv) / sum : 0;
  return best;
}

// Merge four children into their parent. Design rule: nothing safety relevant may get smaller.
//   occupancy  q adds (p = 1 - prod(1-p_i))        heights  absolute max top / min clearance kept
//   ground     evidence weighted mean, the spread goes to grange (kerbs / holes stay visible)
//   semantics  evidence sums                       flags    OR (dynamic / negative survive)
Cell merge4(const Cell* ch) {
  Cell p; std::memset(&p, 0, sizeof p); p.clear = 255;
  int nobs = 0; long gs = 0, gw = 0; int gmin = 32767, gmax = -32768; bool measured = false;
  for (int i = 0; i < 4; i++) if ((ch[i].flags & TF_C_OBSERVED) && (ch[i].flags & TF_C_GROUND)) measured = true;
  for (int i = 0; i < 4; i++) if (ch[i].flags & TF_C_OBSERVED) {
    nobs++;
    // a child whose ground was never hit only carries a guess: it must not pull a measured ground height
    if (measured && !(ch[i].flags & TF_C_GROUND)) continue;
    int w = std::max<int>(1, ch[i].gcnt);
    gs += (long)ch[i].g * w; gw += w; gmin = std::min<int>(gmin, ch[i].g); gmax = std::max<int>(gmax, ch[i].g);
  }
  if (!nobs) return p;
  p.g = (int16_t)std::lround((double)gs / gw);
  int topabs = -100000, clrabs = 100000, occ = 0, gr = gmax - gmin, inh = TF_C_INHERITED, hs[8] = {0};
  int age = 0; uint8_t os = 0; bool first = true; int gc = 0;
  for (int i = 0; i < 4; i++) {
    const Cell& c = ch[i]; if (!(c.flags & TF_C_OBSERVED)) continue;
    if (c.top) topabs = std::max(topabs, c.g + 2 * c.top);
    if (c.clear != 255) clrabs = std::min(clrabs, c.g + 2 * c.clear);
    occ += c.occ; gr = std::max<int>(gr, c.grange); p.neg = std::max(p.neg, c.neg);
    p.flags |= c.flags & (TF_C_OBSERVED | TF_C_GROUND | TF_C_DYNAMIC | TF_C_NEGATIVE);
    inh &= c.flags; gc += c.gcnt;
    for (int k = 0; k < 8; k++) hs[k] += hget(c.hist, k);
    age = std::max<int>(age, c.age); if (first || c.occ) { os = c.ostamp; first = false; }
  }
  p.flags |= inh & TF_C_INHERITED;
  if (topabs > -100000) p.top = (uint8_t)std::min(255, std::max(1, (topabs - p.g + 1) / 2));
  if (clrabs < 100000) p.clear = (uint8_t)std::min(254, std::max(0, (clrabs - p.g) / 2));
  p.occ = (uint8_t)std::min(255, occ); p.grange = (uint8_t)std::min(255, gr);
  p.age = (uint8_t)age; p.ostamp = os; p.gcnt = (uint8_t)std::min(255, gc);
  int mx = 0; for (int k = 0; k < 8; k++) mx = std::max(mx, hs[k]);
  while (mx > 15) { for (int k = 0; k < 8; k++) hs[k] >>= 1; mx >>= 1; }
  for (int k = 0; k < 8; k++) p.hist |= (uint32_t)hs[k] << (4 * k);
  return p;
}

// Split a parent into four children. Occupancy evidence is divided exactly (sum of children ==
// parent, i.e. p_child = 1-(1-p)^(1/4)); heights and hazards are copied (conservative), semantic
// confidence is halved and the children are tagged INHERITED until they are measured directly.
void split4(const Cell& p, Cell* ch) {
  for (int i = 0; i < 4; i++) {
    ch[i] = p;
    if (!(p.flags & TF_C_OBSERVED)) continue;
    ch[i].flags |= TF_C_INHERITED;
    ch[i].occ = (uint8_t)(p.occ / 4 + (i < p.occ % 4 ? 1 : 0));
    if (p.top && !ch[i].occ) ch[i].occ = 0;
    uint32_t h = 0;
    for (int k = 0; k < 8; k++) h |= (uint32_t)((hget(p.hist, k) + 1) >> 1) << (4 * k);
    ch[i].hist = h; ch[i].gcnt = std::min<uint8_t>(p.gcnt, 2);
  }
}

// ------------------------------------------------------------------------------- tile
enum { T_OBST = 1, T_DYN = 2, T_NEG = 4, T_LOWCLEAR = 8, T_STEP = 16, T_FLAT = 32, T_DIRTY = 64 };
struct Tile {
  int32_t tx, ty;
  Cell* cells;
  uint32_t exp_hash;
  uint16_t stamp;
  uint8_t level, risk, want, flags, used, pad;
};

void resample(Tile& t, int target) {
  while (t.level < target) {                       // coarsen
    int n = ncell(t.level + 1);
    Cell* nc = (Cell*)std::malloc(sizeof(Cell) * n);
    for (int i = 0; i < n; i++) nc[i] = merge4(t.cells + 4 * i);
    std::free(t.cells); t.cells = nc; t.level++;
  }
  while (t.level > target) {                       // refine
    int n = ncell(t.level);
    Cell* nc = (Cell*)std::malloc(sizeof(Cell) * n * 4);
    for (int i = 0; i < n; i++) split4(t.cells[i], nc + 4 * i);
    std::free(t.cells); t.cells = nc; t.level--;
  }
}

// ------------------------------------------------------------------------------- network
struct Net {
  int nf = 0, h1 = 0, h2 = 0, h3 = 0, nc = 0; bool ok = false;
  std::vector<float> mu, sd, W1, b1, W2, b2, W3, b3, W4, b4;
};

struct Track {
  int id; double x, y, vx, vy, x0, y0; float ext; int votes[2]; int hits, miss, slow; bool dyn, path; float mscore; float hb[8][4]; int hn;
};

struct Pillar {
  float hmin, hmax, otop, obot;      // obot = bottom of the overhead layer (1e9 = none)
  uint32_t mask; uint16_t cnt, below; uint16_t v[8]; int32_t lidx; int32_t cluster; uint8_t dyn, obst;
  int32_t lowi; float lowh;          // lowest return above 0.25 m (candidate underside of an overhang)
};

struct Cfg {
  int policy = 2, uniform_level = 0, selective_ai = 1, ray_clear = 1, threads = 0, use_net = 1;
  double r0 = 10, r1 = 20, r2 = 40, r3 = 100, max_range = 100;
  double veh_height = 2.0, veh_width = 2.0, rough_thr = 0.010, vres = 0.0074;
};

inline double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

} // namespace

// =============================================================================== engine
struct TFEngine {
  Cfg cfg; Net net;
  std::vector<int32_t> dir; std::vector<Tile> tiles; std::vector<int32_t> freelist;
  std::vector<std::pair<int32_t, int32_t>> removed;
  int frame = 0; double stats[TF_S_COUNT]; std::string decision = "{}";
  // ego
  double ex = 0, ey = 0, ez = 0, eyaw = 0, evx = 0, evy = 0, last_t = -1; bool have_pose = false;
  int otx = 0, oty = 0;
  // per point
  int n = 0;
  std::vector<float> wx, wy, wz, rng, ph, inten; std::vector<uint8_t> pcls, pflag, pconf;
  std::vector<int32_t> ppil, ptile;
  // frame-local tiles
  std::vector<float> tzmin, tg, tgsum, trough, trisk; std::vector<int32_t> tcnt, tgn;
  std::vector<double> ts1, ts2; std::vector<uint16_t> tobst, tai, tlow; std::vector<uint8_t> tacc; std::vector<int32_t> ttouched;
  // frame-local pillars
  std::vector<int32_t> pslot; std::vector<Pillar> pil;
  std::vector<int32_t> g1slot, g2slot, g1t, g2t;
  std::vector<Track> tracks; int next_id = 1;
  long cleared = 0, splits = 0, merges = 0;
  // range image (max range per elevation row x azimuth column) for free-space proofs
  std::vector<float> rimax; std::vector<uint16_t> pcol; std::vector<uint8_t> prow; double Rm[9];
  std::vector<uint8_t> rc_live; std::vector<uint16_t> rc_sub; std::vector<std::vector<std::pair<Tile*, Cell*>>> rc_hits;   // ray-clearing scratch

  TFEngine() : dir(DIRN * DIRN, -1), tzmin(WT * WT), tg(WT * WT), tgsum(WT * WT), trough(WT * WT),
               trisk(WT * WT), tcnt(WT * WT), tgn(WT * WT), ts1(WT * WT), ts2(WT * WT), tobst(WT * WT),
               tai(WT * WT), tlow(WT * WT), tacc(WT * WT), pslot((size_t)WP * WP, -1),
               g1slot((size_t)(WT * 2) * (WT * 2), -1), g2slot((size_t)(WT / 2) * (WT / 2), -1) {
    std::memset(stats, 0, sizeof stats);
  }
  ~TFEngine() { for (auto& t : tiles) if (t.used) std::free(t.cells); }

  // ---- range image helpers
  static inline void ri_bin(float x, float y, float z, float r, int& row, int& col) {
    float el = std::asin(std::max(-1.f, std::min(1.f, z / std::max(r, 1e-3f)))), az = std::atan2(y, x);
    row = std::min(RI_ROWS - 1, std::max(0, (int)((el + RI_EL0) / RI_DEL)));
    col = std::min(RI_COLS - 1, std::max(0, (int)((az + 3.14159265f) * (RI_COLS / 6.2831853f))));
  }
  // true if a beam at the bearing of the pillar's lowest elevated return, aimed between the ground and
  // that return, measured a range clearly beyond it
  bool open_below(const Pillar& p) const {
    int i = p.lowi; if (i < 0) return false;
    // ground point under the return, expressed in the sensor frame
    double gx = wx[i], gy = wy[i], gz = (wz[i] - ph[i]) - ez;
    float sx = (float)(Rm[0] * gx + Rm[3] * gy + Rm[6] * gz), sy = (float)(Rm[1] * gx + Rm[4] * gy + Rm[7] * gz), sz = (float)(Rm[2] * gx + Rm[5] * gy + Rm[8] * gz);
    int rg, cg; ri_bin(sx, sy, sz, std::sqrt(sx * sx + sy * sy + sz * sz), rg, cg);
    int rp = prow[i], c = pcol[i]; float need = rng[i] + 1.0f;
    for (int r = rg + 1; r < rp; r++) if (rimax[(size_t)r * RI_COLS + c] > need) return true;
    return false;
  }

  // ---- directory
  Tile* find(int tx, int ty) {
    int s = dir[wrap(ty) * DIRN + wrap(tx)];
    if (s < 0) return nullptr;
    Tile& t = tiles[s];
    return (t.tx == tx && t.ty == ty) ? &t : nullptr;
  }
  void drop(int slot) {
    Tile& t = tiles[slot];
    if (t.exp_hash) removed.push_back({t.tx, t.ty});
    dir[wrap(t.ty) * DIRN + wrap(t.tx)] = -1; std::free(t.cells); t.cells = nullptr; t.used = 0;
    freelist.push_back(slot);
  }
  Tile* create(int tx, int ty, int level) {
    int d = wrap(ty) * DIRN + wrap(tx);
    if (dir[d] >= 0) drop(dir[d]);                      // stale tile from the far side of the torus
    int s;
    if (!freelist.empty()) { s = freelist.back(); freelist.pop_back(); }
    else { s = (int)tiles.size(); tiles.push_back(Tile()); }
    Tile& t = tiles[s]; std::memset(&t, 0, sizeof t);
    t.tx = tx; t.ty = ty; t.level = (uint8_t)level; t.used = 1;
    t.cells = (Cell*)std::calloc(ncell(level), sizeof(Cell));
    for (int i = 0; i < ncell(level); i++) t.cells[i].clear = 255;
    dir[d] = s; return &t;
  }

  // ---- resolution policy
  inline bool in_window(int tx, int ty) const { int a = tx - otx, b = ty - oty; return a >= 0 && b >= 0 && a < WT && b < WT; }
  double corridor_len() const { double v = std::hypot(evx, evy); return std::min(60.0, std::max(15.0, v * 4.0 + 10.0)); }
  // position in the ego frame (x forward, y left)
  inline void to_ego(double x, double y, double& fx, double& fy) const {
    double c = std::cos(eyaw), s = std::sin(eyaw), dx = x - ex, dy = y - ey;
    fx = c * dx + s * dy; fy = -s * dx + c * dy;
  }
  double tile_risk(int tx, int ty, int tflags) const {
    double cx = (tx + 0.5) * TILE, cy = (ty + 0.5) * TILE, fx, fy, r = 0;
    to_ego(cx, cy, fx, fy);
    double L = corridor_len();
    if (fx > -1.0 && fx < L && std::fabs(fy) < cfg.veh_width * 0.5 + 0.6) r = 0.6 - 0.25 * std::max(0.0, fx) / L;
    if (in_window(tx, ty)) {
      int lt = (ty - oty) * WT + (tx - otx);
      r = std::max(r, (double)trisk[lt]);
      if (tai[lt] >= 3 && tlow[lt] * 10 >= tai[lt] * 3) r = std::max(r, 0.45);        // uncertain semantics
    }
    if (tflags & T_NEG) r = std::max(r, 0.8);
    if (tflags & T_LOWCLEAR) r = std::max(r, 0.6);
    if (tflags & T_STEP) r = std::max(r, 0.4);
    return r;
  }
  int dist_level(int tx, int ty) const {
    double x0 = tx * TILE, y0 = ty * TILE;
    double dx = std::max({x0 - ex, 0.0, ex - (x0 + TILE)}), dy = std::max({y0 - ey, 0.0, ey - (y0 + TILE)});
    double d = std::hypot(dx, dy);
    return d < cfg.r0 ? 0 : d < cfg.r1 ? 1 : d < cfg.r2 ? 2 : d <= cfg.r3 ? 3 : 4;
  }
  int target_level(int tx, int ty, int tflags, double* risk_out = nullptr) const {
    if (cfg.policy == 0) return cfg.uniform_level;
    int ld = dist_level(tx, ty);
    if (cfg.policy == 1) return ld;
    double r = tile_risk(tx, ty, tflags);
    if (risk_out) *risk_out = r;
    int boost = r >= 0.75 ? 2 : r >= 0.35 ? 1 : 0;
    int relax = (r < 0.05 && ld >= 1 && (tflags & T_FLAT) && !(tflags & (T_OBST | T_DYN))) ? 1 : 0;
    return std::min(LMAX, std::max(0, ld - boost + relax));
  }

  void stage_ground(const float* pts, const double* P);
  void stage_triage(float* feats_out, int32_t* g1o, int32_t* g2o, uint8_t* tri_out);
  void stage_net();
  void stage_track(double dt);
  void stage_risk();
  void stage_insert();
  void stage_rayclear();
  void stage_decision();
  void reset_frame();
  void features_for(int i, float* f) const;
};

// ------------------------------------------------------------------------------- 1 ground
void TFEngine::stage_ground(const float* pts, const double* P) {
  double px = P[3], py = P[7], pz = P[11];
  ex = px; ey = py; ez = pz; eyaw = std::atan2(P[4], P[0]);
  otx = floordiv(ex, TILE) - WT / 2; oty = floordiv(ey, TILE) - WT / 2;
  wx.resize(n); wy.resize(n); wz.resize(n); rng.resize(n); ph.resize(n); inten.resize(n);
  pcls.assign(n, 0); pflag.assign(n, 0); pconf.assign(n, 0); ppil.resize(n); ptile.resize(n);
  pcol.resize(n); prow.resize(n); rimax.assign((size_t)RI_ROWS * RI_COLS, 0.f);
  for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) Rm[a * 3 + b] = P[a * 4 + b];
  for (int i = 0; i < n; i++) {
    const float* p = pts + 4 * i;
    wx[i] = (float)(P[0] * p[0] + P[1] * p[1] + P[2] * p[2] + px - ex);      // world-aligned, ego-centred
    wy[i] = (float)(P[4] * p[0] + P[5] * p[1] + P[6] * p[2] + py - ey);
    wz[i] = (float)(P[8] * p[0] + P[9] * p[1] + P[10] * p[2] + pz);
    rng[i] = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]); inten[i] = p[3];
    { int rr, cc; ri_bin(p[0], p[1], p[2], rng[i], rr, cc); prow[i] = (uint8_t)rr; pcol[i] = (uint16_t)cc;
      float& m = rimax[(size_t)rr * RI_COLS + cc]; if (rng[i] > m && std::isfinite(rng[i])) m = rng[i]; }
    int tx = floordiv(wx[i] + ex, TILE) - otx, ty = floordiv(wy[i] + ey, TILE) - oty;
    if (tx < 0 || ty < 0 || tx >= WT || ty >= WT || rng[i] > cfg.max_range || !std::isfinite(wz[i])) { ptile[i] = -1; continue; }
    int lt = ty * WT + tx; ptile[i] = lt;
    if (tcnt[lt] == 0) { ttouched.push_back(lt); tzmin[lt] = wz[i]; }
    else if (wz[i] < tzmin[lt]) tzmin[lt] = wz[i];
    tcnt[lt]++;
  }
  // Ground is grown outward from the vehicle: a tile minimum is accepted as ground only if it is
  // consistent (bounded slope) with ground already accepted nearer to the sensor. This rejects
  // elevated surfaces seen without any ground return (bridge decks, canopies at long range) and
  // holes (tile minimum far below the surrounding surface) in one rule.
  std::vector<std::pair<float, int>> order; order.reserve(ttouched.size()); std::vector<float> nearz;
  for (int lt : ttouched) { double cx = (lt % WT + otx + 0.5) * TILE - ex, cy = (lt / WT + oty + 0.5) * TILE - ey; float d = (float)std::hypot(cx, cy);
    order.push_back({d, lt}); if (d > 2.5f && d < 9.f) nearz.push_back(tzmin[lt]); }
  std::sort(order.begin(), order.end());
  float zseed = (float)ez - 1.8f;
  if (nearz.size() >= 8) { std::nth_element(nearz.begin(), nearz.begin() + nearz.size() / 2, nearz.end()); zseed = nearz[nearz.size() / 2]; }
  for (auto& od : order) {
    int lt = od.second, tx = lt % WT, ty = lt / WT; float d = od.first; double sw = 0, sz = 0, nd = 1e9;
    for (int rad : {2, 5, 10}) {
      for (int dy = -rad; dy <= rad; dy++) for (int dx = -rad; dx <= rad; dx++) {
        int x = tx + dx, y = ty + dy; if (x < 0 || y < 0 || x >= WT || y >= WT) continue; int q = y * WT + x; if (!tacc[q]) continue;
        double dd = std::hypot(dx, dy) * TILE, w = 1.0 / (dd * dd + 0.3); sw += w; sz += w * tg[q]; nd = std::min(nd, dd); }
      if (sw > 0) break;
    }
    float pred;
    if (sw > 0) pred = (float)(sz / sw);
    else if (d < 9.f) { pred = zseed; nd = 0.8; }
    else {                                               // walk towards the vehicle until accepted ground is met
      double ux = (WT / 2 + 0.5 - (tx + 0.5)), uy = (WT / 2 + 0.5 - (ty + 0.5)), ul = std::hypot(ux, uy); ux /= ul; uy /= ul; bool found = false; pred = 0;
      for (int k = 8; k < ul && !found; k++) { int xc = (int)(tx + 0.5 + ux * k), yc = (int)(ty + 0.5 + uy * k);
        for (int j = 0; j < 9 && !found; j++) { int x = xc + j % 3 - 1, y = yc + j / 3 - 1; if (x < 0 || y < 0 || x >= WT || y >= WT) continue;
          int q = y * WT + x; if (tacc[q]) { pred = tg[q]; nd = k * TILE; found = true; } } }
      if (!found) { pred = zseed; nd = d; }
    }
    float tol = (float)std::min(1.0, 0.18 + 0.10 * nd), dz = tzmin[lt] - pred, lo = std::max(0.12f, 0.6f * tol);
    // A tile whose lowest return sits above the prediction is only ground if no adjacent tile shows
    // plausible ground clearly below it. Otherwise it is the lower edge of an object (a bumper, a
    // wall base) and accepting it would lift the ground model and turn the real road into a "hole".
    bool lifted = false;
    if (dz > 0.10f && dz <= tol)
      for (int j = 0; j < 9 && !lifted; j++) { int x = tx + j % 3 - 1, y = ty + j / 3 - 1; if (j == 4 || x < 0 || y < 0 || x >= WT || y >= WT) continue;
        int q = y * WT + x; if (tcnt[q] > 0 && tzmin[q] < tzmin[lt] - 0.18f && tzmin[q] > pred - lo) lifted = true; }
    if (dz > tol || dz < -lo || lifted) tg[lt] = pred;                         // no ground seen here / hole
    else { tg[lt] = tzmin[lt]; tacc[lt] = 1; }
  }
  // refinement: mean of the returns that lie on that reference surface
  for (int i = 0; i < n; i++) { int lt = ptile[i]; if (lt < 0) continue;
    if (std::fabs(wz[i] - tg[lt]) < 0.12f) { tgsum[lt] += wz[i]; tgn[lt]++; } }
  for (int lt : ttouched) if (tgn[lt] >= 4) tg[lt] = tgsum[lt] / tgn[lt];
  // height above ground with bilinear interpolation between tile centres
  for (int i = 0; i < n; i++) {
    int lt = ptile[i]; if (lt < 0) { ph[i] = 0; continue; }
    double fx = (wx[i] + ex) / TILE - otx - 0.5, fy = (wy[i] + ey) / TILE - oty - 0.5;
    int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy); double ax = fx - x0, ay = fy - y0, s = 0, w = 0;
    for (int j = 0; j < 4; j++) {
      int x = x0 + (j & 1), y = y0 + (j >> 1); if (x < 0 || y < 0 || x >= WT || y >= WT) continue;
      int q = y * WT + x; if (tcnt[q] <= 0) continue;
      double ww = ((j & 1) ? ax : 1 - ax) * ((j >> 1) ? ay : 1 - ay) + 1e-4; s += ww * tg[q]; w += ww;
    }
    ph[i] = wz[i] - (float)(w > 0 ? s / w : tg[lt]);
  }
}

// ------------------------------------------------------------------------------- 2 triage
void TFEngine::features_for(int i, float* f) const {
  const Pillar& p = pil[ppil[i]]; int lt = ptile[i];
  auto cl = [](float v, float a, float b) { return v < a ? a : v > b ? b : v; };
  f[0] = cl(ph[i], -1, 4); f[1] = rng[i] / 50.f; f[2] = inten[i];
  f[3] = cl(p.hmin, -1, 4); f[4] = cl(p.hmax, -1, 4); f[5] = f[4] - f[3];
  f[6] = std::log1p((float)p.cnt) / 4.f; f[7] = trough[lt] < 0 ? 1.f : cl(trough[lt], 0, 0.2f) * 10.f;
  f[8] = (float)popcnt(p.mask) / 16.f; f[9] = cl(ph[i] - p.hmin, 0, 4);
  f[10] = std::min<int>(16, tobst[lt]) / 16.f; f[11] = ph[i] >= p.obot - 0.01f ? 1.f : 0.f;
}

void TFEngine::stage_triage(float* feats_out, int32_t* g1o, int32_t* g2o, uint8_t* tri_out) {
  // pillar statistics
  for (int i = 0; i < n; i++) {
    int lt = ptile[i]; if (lt < 0) { ppil[i] = -1; continue; }
    int pxi = floordiv(wx[i] + ex, PILLAR) - otx * 4, pyi = floordiv(wy[i] + ey, PILLAR) - oty * 4;
    pxi = std::min(WP - 1, std::max(0, pxi)); pyi = std::min(WP - 1, std::max(0, pyi));
    int li = pyi * WP + pxi; int s = pslot[li];
    if (s < 0) { s = (int)pil.size(); pslot[li] = s; Pillar q; std::memset(&q, 0, sizeof q);
      q.hmin = 1e9f; q.hmax = -1e9f; q.obot = 1e9f; q.lidx = li; q.cluster = -1; q.lowh = 1e9f; q.lowi = -1; pil.push_back(q); }
    Pillar& p = pil[s]; ppil[i] = s; float h = ph[i];
    p.hmin = std::min(p.hmin, h); p.hmax = std::max(p.hmax, h); if (p.cnt < 65535) p.cnt++;
    int b = (int)((h + 0.5f) / 0.125f); b = b < 0 ? 0 : b > 31 ? 31 : b; p.mask |= 1u << b;
    if (h < -0.12f && p.below < 65535) p.below++;
    if (h > 0.25f && h < p.lowh) { p.lowh = h; p.lowi = i; }
    if (h <= 0.10f && h >= -0.12f) { ts1[lt] += h; ts2[lt] += (double)h * h; }
  }
  // tile roughness from ground-like returns (tgn is reused as a counter below)
  std::vector<int32_t>& gl = tgn; for (int lt : ttouched) gl[lt] = 0;
  for (int i = 0; i < n; i++) { int lt = ptile[i]; if (lt >= 0 && ph[i] <= 0.10f && ph[i] >= -0.12f) gl[lt]++; }
  for (int lt : ttouched) {
    if (gl[lt] >= 5) { double m = ts1[lt] / gl[lt]; trough[lt] = (float)std::sqrt(std::max(0.0, ts2[lt] / gl[lt] - m * m)); }
    else trough[lt] = -1;
  }
  // vertical structure of every pillar: obstacle column from the ground, and overhead layer
  for (Pillar& p : pil) {
    int lx = p.lidx % WP, ly = p.lidx / WP;
    double r = std::hypot((lx + 0.5) * PILLAR + otx * TILE - ex, (ly + 0.5) * PILLAR + oty * TILE - ey);
    int tol = (int)std::ceil(std::max(0.5, 3.0 * r * cfg.vres) / 0.125);
    uint32_t m = p.mask >> 6;                         // bins above 0.25 m
    if (p.hmax > 0.15f) p.obst = 1;
    if (!m) { p.otop = p.hmax > 0.15f ? p.hmax : 0; continue; }
    int first = ctz32(m);
    float hfirst = 0.25f + 0.125f * first;
    // Overhead structure only if a lower beam on the same bearing went on past this range, i.e. the
    // space beneath is observed free. Otherwise it is a normal obstacle whose lower part was not sampled
    // (a far vehicle seen by one beam), and treating it as passable would be the unsafe choice.
    if (hfirst >= std::max(0.8f, 0.125f * tol) && !(p.mask & 0x30u) && open_below(p)) { p.obot = hfirst; p.otop = 0; p.obst = 0; continue; }
    int last = first, gap = 0, b = first + 1;
    for (; b < 26; b++) { if ((m >> b) & 1) { if (gap >= tol) break; last = b; gap = 0; } else gap++; }
    p.otop = std::min(p.hmax, 0.25f + 0.125f * (last + 1));
    if (b < 26 && ((m >> b) & 1)) p.obot = 0.25f + 0.125f * b;
  }
  for (const Pillar& p : pil) if (p.obst) { int lt = (p.lidx / WP / 4) * WT + (p.lidx % WP) / 4; if (tobst[lt] < 65535) tobst[lt]++; }
  // decide per point
  for (int i = 0; i < n; i++) {
    int lt = ptile[i]; if (lt < 0) { if (tri_out) tri_out[i] = 255; continue; }
    float h = ph[i]; const Pillar& p = pil[ppil[i]]; uint8_t tri = 1;
    if (h >= p.obot - 0.01f) pflag[i] |= TF_P_OVERHANG;
    if (h < -0.12f) { pcls[i] = TF_TERRAIN; pconf[i] = 200; pflag[i] |= TF_P_NEGATIVE; tri = 2; }
    else if (h <= 0.10f) {
      pflag[i] |= TF_P_GROUNDLIKE;
      bool smooth = trough[lt] >= 0 && trough[lt] < cfg.rough_thr;
      if (smooth) { pcls[i] = TF_DRIVABLE; pconf[i] = 230; tri = 0; }
      else { pcls[i] = TF_TERRAIN; pconf[i] = 120; }
    } else {                                           // geometry-only fallback labels
      pcls[i] = (pflag[i] & TF_P_OVERHANG) ? TF_STRUCTURE : (p.otop < 0.3f ? TF_TERRAIN : TF_STRUCTURE); pconf[i] = 90;
    }
    if (tri_out) tri_out[i] = tri;
    if (tri == 1 || (!cfg.selective_ai && tri == 0)) pflag[i] |= TF_P_AI;
    if (feats_out) {
      features_for(i, feats_out + (size_t)TF_NFEAT * i);
      int gx = floordiv(wx[i] + ex, 0.4) - otx * 2, gy = floordiv(wy[i] + ey, 0.4) - oty * 2;
      g1o[i] = std::min(WT * 2 - 1, std::max(0, gy)) * WT * 2 + std::min(WT * 2 - 1, std::max(0, gx));
      g2o[i] = (lt / WT / 2) * (WT / 2) + (lt % WT) / 2;
    }
  }
}

// ------------------------------------------------------------------------------- 3 network
void TFEngine::stage_net() {
  if (!net.ok || !cfg.use_net) return;
  std::vector<int> idx; idx.reserve(n);
  for (int i = 0; i < n; i++) if (pflag[i] & TF_P_AI) idx.push_back(i);
  int m = (int)idx.size(); if (!m) return;
  const int nf = net.nf, h1 = net.h1, h2 = net.h2, h3 = net.h3, nc = net.nc;
  std::vector<float> A2((size_t)m * h2); std::vector<int32_t> s1(m), s2(m);
  int n1 = 0, n2 = 0;
  for (int k = 0; k < m; k++) {
    int i = idx[k], lt = ptile[i];
    int gx = std::min(WT * 2 - 1, std::max(0, floordiv(wx[i] + ex, 0.4) - otx * 2));
    int gy = std::min(WT * 2 - 1, std::max(0, floordiv(wy[i] + ey, 0.4) - oty * 2));
    int a = gy * WT * 2 + gx, b = (lt / WT / 2) * (WT / 2) + (lt % WT) / 2;
    if (g1slot[a] < 0) { g1slot[a] = n1++; g1t.push_back(a); }
    if (g2slot[b] < 0) { g2slot[b] = n2++; g2t.push_back(b); }
    s1[k] = g1slot[a]; s2[k] = g2slot[b];
  }
#pragma omp parallel for schedule(static)
  for (int k = 0; k < m; k++) {
    float f[TF_NFEAT], a1[64];
    features_for(idx[k], f);
    for (int j = 0; j < nf; j++) f[j] = (f[j] - net.mu[j]) / net.sd[j];
    for (int o = 0; o < h1; o++) { float s = net.b1[o]; const float* w = &net.W1[(size_t)o * nf];
      for (int j = 0; j < nf; j++) s += w[j] * f[j]; a1[o] = s > 0 ? s : 0; }
    float* a2 = &A2[(size_t)k * h2];
    for (int o = 0; o < h2; o++) { float s = net.b2[o]; const float* w = &net.W2[(size_t)o * h1];
      for (int j = 0; j < h1; j++) s += w[j] * a1[j]; a2[o] = s > 0 ? s : 0; }
  }
  std::vector<float> P1((size_t)n1 * h2, 0.f), P2((size_t)n2 * h2, 0.f);       // max pooling (ReLU >= 0)
  for (int k = 0; k < m; k++) {
    const float* a2 = &A2[(size_t)k * h2]; float* p1 = &P1[(size_t)s1[k] * h2]; float* p2 = &P2[(size_t)s2[k] * h2];
    for (int j = 0; j < h2; j++) { if (a2[j] > p1[j]) p1[j] = a2[j]; if (a2[j] > p2[j]) p2[j] = a2[j]; }
  }
#pragma omp parallel for schedule(static)
  for (int k = 0; k < m; k++) {
    float z[192], a3[128], lg[TF_NCLS];
    std::memcpy(z, &A2[(size_t)k * h2], sizeof(float) * h2);
    std::memcpy(z + h2, &P1[(size_t)s1[k] * h2], sizeof(float) * h2);
    std::memcpy(z + 2 * h2, &P2[(size_t)s2[k] * h2], sizeof(float) * h2);
    for (int o = 0; o < h3; o++) { float s = net.b3[o]; const float* w = &net.W3[(size_t)o * 3 * h2];
      for (int j = 0; j < 3 * h2; j++) s += w[j] * z[j]; a3[o] = s > 0 ? s : 0; }
    int best = 0; float mx = -1e30f;
    for (int o = 0; o < nc; o++) { float s = net.b4[o]; const float* w = &net.W4[(size_t)o * h3];
      for (int j = 0; j < h3; j++) s += w[j] * a3[j]; lg[o] = s; if (s > mx) { mx = s; best = o; } }
    float den = 0; for (int o = 0; o < nc; o++) den += std::exp(lg[o] - mx);
    int i = idx[k]; pcls[i] = (uint8_t)best; pconf[i] = (uint8_t)(255.f / den);
  }
  for (int a : g1t) g1slot[a] = -1; for (int b : g2t) g2slot[b] = -1; g1t.clear(); g2t.clear();
}

// ------------------------------------------------------------------------------- 4 objects
void TFEngine::stage_track(double dt) {
  // semantic votes per pillar (lower obstacle layer only)
  for (int i = 0; i < n; i++) {
    if (ptile[i] < 0) continue; Pillar& p = pil[ppil[i]];
    if (ph[i] > 0.15f && !(pflag[i] & TF_P_OVERHANG) && p.v[pcls[i]] < 65535) p.v[pcls[i]]++;
    int lt = ptile[i];
    if (pflag[i] & TF_P_AI) { if (tai[lt] < 65535) tai[lt]++; if (pconf[i] < 140 && tlow[lt] < 65535) tlow[lt]++; }
  }
  // What did the map hold at this spot before this sweep?  0 unknown, 1 free ground, 2 recently occupied,
  // 3 solidly occupied (8+ sweeps). A real mover keeps putting returns onto free ground; a parked
  // object seen from a new angle only puts them on unknown or already occupied cells.
  auto prior = [&](double X, double Y) {
    int tx = floordiv(X, TILE), ty = floordiv(Y, TILE); Tile* t = find(tx, ty); if (!t) return 0; int k = cps(t->level); double cs = csize(t->level);
    int cxi = std::min(k - 1, std::max(0, (int)((X - tx * TILE) / cs))), cyi = std::min(k - 1, std::max(0, (int)((Y - ty * TILE) / cs)));
    const Cell& c = t->cells[morton(cxi << t->level, cyi << t->level) >> (2 * t->level)];
    if (c.occ && c.top) return c.age >= 8 ? 3 : 2;
    if (!(c.flags & TF_C_OBSERVED) || (c.flags & TF_C_INHERITED)) return 0;
    return (c.flags & TF_C_GROUND) ? 1 : 0; };
  struct Cl { double sx, sy, x0, x1, y0, y1; int np, veh, per; std::vector<int> pill; int nfree = 0, nold = 0, ntot = 0; };
  std::vector<Cl> cls; std::vector<int> stack;
  // clusters are geometric (any obstacle column); the semantic vote is taken per cluster, which keeps
  // an object in one piece even when single pillars are mislabelled
  auto is_obj = [&](const Pillar& p) { int t = 0; for (int k = 3; k < 8; k++) t += p.v[k]; return p.obst && p.otop > 0.3f && t >= 1; };
  for (int s = 0; s < (int)pil.size(); s++) {
    if (pil[s].cluster >= 0 || !is_obj(pil[s])) continue;
    Cl c{0, 0, 1e18, -1e18, 1e18, -1e18, 0, 0, 0, {}, 0, 0, 0}; int id = (int)cls.size();
    pil[s].cluster = id; stack.push_back(s);
    while (!stack.empty()) {
      int q = stack.back(); stack.pop_back(); Pillar& p = pil[q]; c.pill.push_back(q);
      int lx = p.lidx % WP, ly = p.lidx / WP; double x = (lx + 0.5) * PILLAR + otx * TILE, y = (ly + 0.5) * PILLAR + oty * TILE;
      int w = 0; for (int kk = 3; kk < 8; kk++) w += p.v[kk]; c.sx += x * w; c.sy += y * w; c.np += w; c.veh += p.v[6]; c.per += p.v[7];
      c.x0 = std::min(c.x0, x); c.x1 = std::max(c.x1, x); c.y0 = std::min(c.y0, y); c.y1 = std::max(c.y1, y);
      int rr = std::hypot(x - ex, y - ey) > 25.0 ? 2 : 1;
      for (int dy = -rr; dy <= rr; dy++) for (int dx = -rr; dx <= rr; dx++) {
        int X = lx + dx, Y = ly + dy; if (X < 0 || Y < 0 || X >= WP || Y >= WP) continue;
        int t = pslot[Y * WP + X]; if (t >= 0 && pil[t].cluster < 0 && is_obj(pil[t])) { pil[t].cluster = id; stack.push_back(t); }
      }
    }
    cls.push_back(std::move(c));
  }
  for (int i = 0; i < n; i++) {
    if (ptile[i] < 0 || ph[i] <= 0.15f || (pflag[i] & TF_P_OVERHANG)) continue; int ci = pil[ppil[i]].cluster; if (ci < 0) continue;
    int st = prior(wx[i] + ex, wy[i] + ey); cls[ci].ntot++; if (st == 1) cls[ci].nfree++; else if (st == 3) cls[ci].nold++;
  }
  for (Track& t : tracks) { t.x += t.vx * dt; t.y += t.vy * dt; }
  struct Pair { double d; int c, t; }; std::vector<Pair> pairs;
  std::vector<double> cx(cls.size()), cy(cls.size()), cext(cls.size());
  for (size_t c = 0; c < cls.size(); c++) {
    cx[c] = 0.5 * (cls[c].x0 + cls[c].x1); cy[c] = 0.5 * (cls[c].y0 + cls[c].y1);
    cext[c] = std::max(cls[c].x1 - cls[c].x0, cls[c].y1 - cls[c].y0) + PILLAR;
    if (cls[c].np < 3 || cext[c] > 9.0 || 2 * (cls[c].veh + cls[c].per) < cls[c].np) { cext[c] = 1e9; continue; }
    for (size_t t = 0; t < tracks.size(); t++) {
      double d = std::hypot(cx[c] - tracks[t].x, cy[c] - tracks[t].y);
      if (d < 1.2 + 0.5 * std::max<double>(cext[c], tracks[t].ext)) pairs.push_back({d, (int)c, (int)t});
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.d < b.d; });
  std::vector<int> cm(cls.size(), -1), tm(tracks.size(), -1);
  for (auto& p : pairs) if (cm[p.c] < 0 && tm[p.t] < 0) { cm[p.c] = p.t; tm[p.t] = p.c; }
  for (size_t t = 0; t < tracks.size(); t++) {
    Track& k = tracks[t];
    if (tm[t] < 0) { k.miss++; continue; }
    int c = tm[t]; double rx = cx[c] - k.x, ry = cy[c] - k.y;
    k.x += 0.5 * rx; k.y += 0.5 * ry; k.vx += 0.25 * rx / dt; k.vy += 0.25 * ry / dt;
    k.ext = std::max<float>(k.ext * 0.98f, (float)cext[c]); k.votes[0] += cls[c].veh; k.votes[1] += cls[c].per; k.hits++; k.miss = 0;
    double sp = std::hypot(k.vx, k.vy), disp = std::hypot(k.x - k.x0, k.y - k.y0);
    // Motion test that survives changing viewpoints: a parked object seen from a new angle grows on one
    // side only; a moving object shifts BOTH opposite edges of its footprint the same way. On top of
    // that it must keep entering cells the map knew as free ground.
    float nb[4] = {(float)cls[c].x0, (float)cls[c].x1, (float)cls[c].y0, (float)cls[c].y1};
    const float* ob = k.hb[(k.hn + 2) % 8];                               // footprint 6 sweeps ago
    bool edges = false;
    if (k.hn >= 6) { float a0 = nb[0] - ob[0], a1 = nb[1] - ob[1], b0 = nb[2] - ob[2], b1 = nb[3] - ob[3]; const float th = 0.30f;
      edges = (a0 > th && a1 > th) || (a0 < -th && a1 < -th) || (b0 > th && b1 > th) || (b0 < -th && b1 < -th); }
    std::memcpy(k.hb[k.hn % 8], nb, sizeof nb); k.hn++;
    // and it must not sit on cells that have been solidly occupied for a second or more
    bool settled = cls[c].nold * 10 >= cls[c].ntot * 6;
    bool far = std::hypot(cx[c] - ex, cy[c] - ey) > 20.0;                 // ground is too sparsely sampled out there
    bool ev = edges && !settled && (far || (cls[c].nfree >= 3 && cls[c].nfree * 100 >= cls[c].ntot * 15));
    if (settled) { k.mscore = std::max(0.f, k.mscore - 1.f); k.slow += 3; }
    k.mscore = std::min(8.f, std::max(0.f, k.mscore + (ev ? 1.f : -0.5f)));
    if (!k.dyn && sp > 0.4 && disp > 0.5 && k.mscore >= 3.f) k.dyn = true;
    if (k.dyn) { k.slow = (sp < 0.3 || k.mscore < 1.f) ? k.slow + 1 : 0; if (k.slow > 8) { k.dyn = false; k.x0 = k.x; k.y0 = k.y; } }
  }
  for (size_t c = 0; c < cls.size(); c++) if (cm[c] < 0 && cls[c].np >= 5 && cext[c] <= 9.0)
    tracks.push_back(Track{next_id++, cx[c], cy[c], 0, 0, cx[c], cy[c], (float)cext[c], {cls[c].veh, cls[c].per}, 1, 0, 0, false, false, 0.f, {{0}}, 0});
  tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [&](const Track& t) {
    return t.miss > 6 || std::hypot(t.x - ex, t.y - ey) > cfg.max_range + 5; }), tracks.end());
  // (track vector may have been compacted: re-associate by nearest dynamic track)
  for (size_t c = 0; c < cls.size(); c++) {
    bool dyn = false;
    if (cext[c] > 9.0) continue;
    for (const Track& k : tracks) if (k.dyn && k.miss == 0 && std::hypot(cx[c] - k.x, cy[c] - k.y) < 0.6 + 0.5 * k.ext) { dyn = true; break; }
    if (dyn) for (int q : cls[c].pill) pil[q].dyn = 1;
  }
  for (int i = 0; i < n; i++) if (ptile[i] >= 0 && pil[ppil[i]].dyn && ph[i] > 0.15f && !(pflag[i] & TF_P_OVERHANG)) pflag[i] |= TF_P_DYNAMIC;
}

// ------------------------------------------------------------------------------- 5 risk -> resolution
void TFEngine::stage_risk() {
  double L = corridor_len(), hw = cfg.veh_width * 0.5 + 0.6;
  auto stamp = [&](double x, double y, double rad, float val) {
    int x0 = floordiv(x - rad, TILE) - otx, x1 = floordiv(x + rad, TILE) - otx, y0 = floordiv(y - rad, TILE) - oty, y1 = floordiv(y + rad, TILE) - oty;
    for (int Y = std::max(0, y0); Y <= std::min(WT - 1, y1); Y++) for (int X = std::max(0, x0); X <= std::min(WT - 1, x1); X++) {
      int lt = Y * WT + X; if (trisk[lt] == 0 && tcnt[lt] == 0) ttouched.push_back(lt), tcnt[lt] = -1;
      trisk[lt] = std::max(trisk[lt], val);
    }
  };
  for (Track& k : tracks) {
    if (k.miss > 2) continue; k.path = false;
    double rad = 0.5 * k.ext + 0.8;
    if (!k.dyn) { stamp(k.x, k.y, rad, 0.36f); continue; }       // potential mover
    for (double s = 0; s <= 2.5; s += 0.25) {
      double x = k.x + k.vx * s, y = k.y + k.vy * s, fx, fy; to_ego(x, y, fx, fy);
      if (fx > 0 && fx < L && std::fabs(fy) < hw + 0.5 * k.ext) k.path = true;
    }
    for (double s = 0; s <= 2.5; s += 0.25) stamp(k.x + k.vx * s, k.y + k.vy * s, rad, s == 0 ? 0.9f : (k.path ? 0.9f : 0.5f));
  }
  int boosted = 0, relaxed = 0; uint16_t f16 = (uint16_t)frame;
  for (int s = 0; s < (int)tiles.size(); s++) {
    Tile& t = tiles[s]; if (!t.used) continue;
    double cx = (t.tx + 0.5) * TILE - ex, cy = (t.ty + 0.5) * TILE - ey;
    if (std::max(std::fabs(cx), std::fabs(cy)) > cfg.max_range + TILE || cx * cx + cy * cy > (cfg.max_range + 1.2) * (cfg.max_range + 1.2)) { drop(s); continue; }
    if (t.flags & T_DIRTY) {                                    // refresh hazard summary of the tile
      int fl = 0; int gmin = 32767, gmax = -32768; double climit = (cfg.veh_height + 0.3) * 50.0;
      // Cells that were only ever hit above the ground (under a gantry, behind a kerb) carry a guessed
      // ground height from when they were first seen, possibly at long range. Re-anchor that guess on
      // ground actually measured in the same tile (or this sweep's ground model), keeping the absolute
      // height of whatever is stored above it.
      { long gs = 0; int gn = 0, gref = INT32_MIN;
        for (int i = 0; i < ncell(t.level); i++) { const Cell& c = t.cells[i]; if ((c.flags & TF_C_OBSERVED) && (c.flags & TF_C_GROUND)) { gs += c.g; gn++; } }
        if (gn) gref = (int)(gs / gn);
        else if (in_window(t.tx, t.ty)) { int lt = (t.ty - oty) * WT + (t.tx - otx); if (tacc[lt]) gref = (int)std::lround(tg[lt] * 100.f); }
        if (gref != INT32_MIN) for (int i = 0; i < ncell(t.level); i++) { Cell& c = t.cells[i];
          if (!(c.flags & TF_C_OBSERVED) || (c.flags & TF_C_GROUND)) continue;
          int sh = (c.g - gref) / 2; if (!sh) continue;
          c.g = (int16_t)(c.g - 2 * sh);
          if (c.top) c.top = (uint8_t)std::min(255, std::max(1, c.top + sh));
          if (c.clear != 255) c.clear = (uint8_t)std::min(254, std::max(0, c.clear + sh)); } }
      for (int i = 0; i < ncell(t.level); i++) { const Cell& c = t.cells[i]; if (!(c.flags & TF_C_OBSERVED)) continue;
        if (has_obstacle(c)) fl |= T_OBST; if (c.flags & TF_C_DYNAMIC) fl |= T_DYN; if (c.neg >= 100) fl |= T_NEG;
        if (c.clear != 255 && c.clear < climit) fl |= T_LOWCLEAR;
        if (c.flags & TF_C_GROUND) { gmin = std::min<int>(gmin, c.g); gmax = std::max<int>(gmax, c.g); } }
      if (gmax - gmin >= 10 && gmax - gmin < 60) fl |= T_STEP;
      t.flags = (uint8_t)((t.flags & T_FLAT) | fl);
    }
    if (in_window(t.tx, t.ty)) { int lt = (t.ty - oty) * WT + (t.tx - otx);
      if (tcnt[lt] > 0) { bool flat = trough[lt] >= 0 && trough[lt] < cfg.rough_thr && tobst[lt] == 0; t.flags = (uint8_t)(flat ? t.flags | T_FLAT : t.flags & ~T_FLAT); } }
    double r = 0; int tgt = target_level(t.tx, t.ty, t.flags, &r); t.risk = (uint8_t)(std::min(1.0, r) * 255);
    if (cfg.policy == 2) { int ld = dist_level(t.tx, t.ty); if (tgt < ld) boosted++; else if (tgt > ld) relaxed++; }
    if (tgt < t.level) { splits += 1; resample(t, tgt); t.want = 0; t.flags |= T_DIRTY; }
    else if (tgt > t.level) { if (++t.want >= 5) { merges += 1; resample(t, tgt); t.want = 0; t.flags |= T_DIRTY; } }
    else t.want = 0;
    (void)f16;
  }
  stats[TF_S_BOOSTED_TILES] = boosted; stats[TF_S_RELAXED_TILES] = relaxed;
}

// ------------------------------------------------------------------------------- 6 map update
void TFEngine::stage_insert() {
  uint16_t f16 = (uint16_t)frame; uint8_t f8 = (uint8_t)frame;
  Tile* cache = nullptr; int ctx = INT32_MIN, cty = INT32_MIN;
  for (int i = 0; i < n; i++) {
    int lt = ptile[i]; if (lt < 0) continue;
    double X = wx[i] + ex, Y = wy[i] + ey; int tx = lt % WT + otx, ty = lt / WT + oty;
    if (tx != ctx || ty != cty) {
      cache = find(tx, ty);
      if (!cache) { int fl = (trough[lt] >= 0 && trough[lt] < cfg.rough_thr && tobst[lt] == 0) ? T_FLAT : 0;
        cache = create(tx, ty, target_level(tx, ty, fl)); cache->flags = (uint8_t)fl; }
      ctx = tx; cty = ty;
    }
    Tile& t = *cache; int k = cps(t.level); double cs = csize(t.level);
    int cxi = std::min(k - 1, std::max(0, (int)((X - tx * TILE) / cs))), cyi = std::min(k - 1, std::max(0, (int)((Y - ty * TILE) / cs)));
    Cell& c = t.cells[morton(cxi << t.level, cyi << t.level) >> (2 * t.level)];
    t.stamp = f16; t.flags |= T_DIRTY;
    int zc = (int)std::lround(wz[i] * 100.f); float h = ph[i]; int cl = pcls[i];
    if (!(c.flags & TF_C_OBSERVED)) { c.g = (int16_t)std::lround((wz[i] - h) * 100.f); c.flags |= TF_C_OBSERVED; }
    c.flags &= ~TF_C_INHERITED;
    bool groundish = (cl == TF_DRIVABLE || cl == TF_TERRAIN) && h < 0.4f;
    if (groundish) {
      int g_old = c.g;
      if (!(c.flags & TF_C_GROUND)) { c.g = (int16_t)zc; c.gcnt = 1; c.flags |= TF_C_GROUND; c.grange = 0; }
      else { int d = zc - c.g;
        if (std::abs(d) > 50 && c.gcnt >= 3) continue;         // isolated outlier against an established ground height
        c.grange = (uint8_t)std::min(255, std::max<int>(c.grange, std::abs(d)));
        if (c.gcnt < 12) c.gcnt++; c.g = (int16_t)(c.g + d / (int)c.gcnt); }
      // top and clearance are stored relative to the ground: keep their absolute height when the
      // ground reference changes (typically a far-range guess replaced by the first real ground return)
      if (int sh = (g_old - c.g) / 2) {
        if (c.top) c.top = (uint8_t)std::min(255, std::max(1, c.top + sh));
        if (c.clear != 255) c.clear = (uint8_t)std::min(254, std::max(0, c.clear + sh)); }
      if (pflag[i] & TF_P_NEGATIVE) {                          // below the local ground surface
        const Pillar& p = pil[ppil[i]]; int add = std::min(70, std::max(12, (int)(-h * 220.f))) * std::min<int>(3, p.below) / 3;
        c.neg = (uint8_t)std::min(255, c.neg + add); if (c.neg >= 100) c.flags |= TF_C_NEGATIVE;
      } else if (c.neg) { c.neg = (uint8_t)(c.neg > 6 ? c.neg - 6 : 0); if (c.neg < 60) c.flags &= ~TF_C_NEGATIVE; }
      hadd(c.hist, cl, pconf[i] > 160 ? 2 : 1);
    } else if (h > 0.15f) {
      float hc = (wz[i] - c.g * 0.01f);                        // height above the cell's own ground
      // only static things can be an overhead structure: an elevated return on a vehicle or a person
      // is stored as an obstacle, so it is tracked and ray-cleared instead of leaving a permanent
      // low-clearance mark along the path of a moving object
      bool overhead = (pflag[i] & TF_P_OVERHANG) && !(pflag[i] & TF_P_DYNAMIC) && cl != TF_VEHICLE && cl != TF_PERSON;
      if (overhead) { int u = std::min(254, std::max(0, (int)(hc * 50.f))); if (u < c.clear) c.clear = (uint8_t)u; }
      else {
        int u = std::min(255, std::max(1, (int)std::ceil(hc * 50.f)));
        if (c.ostamp != f8 || !c.occ) {                         // first return of this sweep in the cell
          if (!c.occ || u > c.top) c.top = (uint8_t)u; c.ostamp = f8; if (!(pflag[i] & TF_P_DYNAMIC)) c.flags &= ~TF_C_DYNAMIC;
          c.occ = (uint8_t)std::min(OCC_MAX, c.occ + OCC_HIT); if (c.age < 255) c.age++; }
        else if (u > c.top) c.top = (uint8_t)u;
        if (pflag[i] & TF_P_DYNAMIC) c.flags |= TF_C_DYNAMIC;
      }
      hadd(c.hist, cl, pconf[i] > 160 ? 2 : 1);
    }
  }
}

// ------------------------------------------------------------------------------- 7 ray clearing
// Every ray is proof of free space between the sensor and its return. If such a ray crosses a
// stored obstacle column below its top, that obstacle is no longer there (it moved away).
void TFEngine::stage_rayclear() {
  if (!cfg.ray_clear) return;
  uint8_t f8 = (uint8_t)frame; long cl = 0;
  // Tiles that hold anything clearable, as a flat window bitmap: the march below tests one byte
  // per step instead of a directory lookup, so rays over empty ground cost almost nothing.
  // Second level: a 4x4 bitmap per tile (20 cm blocks) of where a clearable obstacle cell actually
  // sits (stored, has height, not confirmed by this sweep). Rays only step finely inside set blocks.
  rc_live.assign((size_t)WT * WT, 0); rc_sub.assign(tiles.size(), 0);
  for (size_t ti = 0; ti < tiles.size(); ti++) {
    const Tile& t = tiles[ti];
    if (!t.used || !(t.flags & (T_OBST | T_DIRTY)) || !in_window(t.tx, t.ty)) continue;
    int k = cps(t.level); uint16_t m = 0;
    for (int cy = 0; cy < k; cy++) for (int cx = 0; cx < k; cx++) {
      const Cell& c = t.cells[morton(cx << t.level, cy << t.level) >> (2 * t.level)];
      if (!c.occ || !c.top || c.ostamp == f8) continue;
      int x0 = cx * 4 / k, x1 = ((cx + 1) * 4 - 1) / k, y0 = cy * 4 / k, y1 = ((cy + 1) * 4 - 1) / k;
      for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) m |= (uint16_t)(1u << (y * 4 + x));
    }
    rc_sub[ti] = m;
    if (m) rc_live[(size_t)(t.ty - oty) * WT + (t.tx - otx)] = 1;
  }
  const Tile* tile0 = tiles.data(); const double sub_inv = 4.0 / TILE;
  const double inv_tile = 1.0 / TILE, wox = otx * TILE, woy = oty * TILE;
  const uint8_t* live_map = rc_live.data();
  int nthr = std::max(1, tf_max_threads());
  if ((int)rc_hits.size() < nthr) rc_hits.resize(nthr);
  for (auto& h : rc_hits) h.clear();
  // Phase 1 (parallel, read-only): collect every stored obstacle cell a ray passed through.
#pragma omp parallel
  {
    auto& H = rc_hits[tf_thread_id()];
#pragma omp for schedule(static)
    for (int i = 0; i < n; i += 3) {
      if (ptile[i] < 0 || rng[i] < 3.f) continue;
      double X1 = wx[i] + ex, Y1 = wy[i] + ey, Z1 = wz[i], dx = X1 - ex, dy = Y1 - ey, dz = Z1 - ez;
      double len = std::hypot(dx, dy); if (len < 2.0) continue;
      double stop = len - 0.45, step = 0.4, il = 1.0 / len;
      auto live_at = [&](double q) -> bool { double uu = q * il; int a = (int)((ex + dx * uu - wox) * inv_tile), b = (int)((ey + dy * uu - woy) * inv_tile);
        return a >= 0 && b >= 0 && a < WT && b < WT && live_map[(size_t)b * WT + a]; };
      int ltx = INT32_MIN, lty = INT32_MIN; Tile* t = nullptr;
      auto tile_at = [&](double q) -> Tile* { double uu = q * il; int tx = floordiv(ex + dx * uu, TILE), ty = floordiv(ey + dy * uu, TILE);
        if (tx != ltx || ty != lty) { t = find(tx, ty); ltx = tx; lty = ty; } return t; };
      auto live = [](const Tile* q) { return q && (q->flags & (T_OBST | T_DIRTY)); };
      bool prev = live_at(1.6);
      for (double s = 1.6; s < stop; s += step) {
        bool next = live_at(std::min(stop, s + step)), cur = prev; prev = next;
        if (!cur && !next) continue;                                           // nothing stored along this stretch
        double q = s, qe = std::min(stop, s + step);
        while (q < qe) {
          Tile* tt = tile_at(q); if (!live(tt)) { q += CELL0 * 2; continue; }
          double cs = csize(tt->level), uu = q * il, xx = ex + dx * uu, yy = ey + dy * uu, zz = ez + dz * uu;
          int sx = std::min(3, std::max(0, (int)((xx - tt->tx * TILE) * sub_inv))), sy = std::min(3, std::max(0, (int)((yy - tt->ty * TILE) * sub_inv)));
          if (!((rc_sub[tt - tile0] >> (sy * 4 + sx)) & 1)) { q += 0.1; continue; }      // empty 20 cm block
          int k = cps(tt->level);
          int cxi = std::min(k - 1, std::max(0, (int)((xx - tt->tx * TILE) / cs))), cyi = std::min(k - 1, std::max(0, (int)((yy - tt->ty * TILE) / cs)));
          Cell& c = tt->cells[morton(cxi << tt->level, cyi << tt->level) >> (2 * tt->level)];
          q += std::min(step, cs);
          if (!c.occ || !c.top || c.ostamp == f8) continue;                    // nothing stored, or confirmed this sweep
          double hr = zz - c.g * 0.01;
          if (hr < 0.12 || hr > c.top * 0.02 - 0.06) continue;                 // ray passed over it: no evidence
          // one ray vouches for a 5 cm cell, but only for a sliver of a 40 cm cell: coarse cells need
          // more rays and must have gone un-hit for longer before they are released
          if ((uint8_t)(f8 - c.ostamp) < 1 + 2 * tt->level) continue;
          H.push_back({tt, &c});
        }
      }
    }
  }
  // Phase 2 (serial, in ray order): apply the evidence. Same result as a single-threaded march.
  for (auto& H : rc_hits) for (auto& hc : H) {
    Tile* tt = hc.first; Cell& c = *hc.second;
    if (!c.occ || !c.top) continue;
    int lv = tt->level, dec = std::max(1, OCC_CLEAR >> lv); if (c.flags & TF_C_DYNAMIC) dec *= 4;
    if (c.occ <= dec) { c.occ = 0; c.top = 0; c.age = 0; c.flags &= ~TF_C_DYNAMIC; cl++; tt->flags |= T_DIRTY;
      uint32_t h = 0; for (int kk = 1; kk <= 2; kk++) h |= (uint32_t)hget(c.hist, kk) << (4 * kk); c.hist = h; }
    else c.occ = (uint8_t)(c.occ - dec);
  }
  cleared += cl; stats[TF_S_CLEARED] = (double)cl;
}

// ------------------------------------------------------------------------------- 8 decision
void TFEngine::stage_decision() {
  double L = corridor_len(), hw = cfg.veh_width * 0.5 + 0.2, v = std::hypot(evx, evy);
  double c = std::cos(eyaw), s = std::sin(eyaw);
  double best = 1e9; int kind = 0, kcls = 0; double kval = 0; int cnt[4] = {0, 0, 0, 0}; double dist[4] = {1e9, 1e9, 1e9, 1e9}; double val[4] = {0, 0, 0, 0}; int kc[4] = {0, 0, 0, 0};
  int x0 = floordiv(ex - L - 2, TILE), x1 = floordiv(ex + L + 2, TILE), y0 = floordiv(ey - L - 2, TILE), y1 = floordiv(ey + L + 2, TILE);
  int negc = 0;
  for (int ty = y0; ty <= y1; ty++) for (int tx = x0; tx <= x1; tx++) {
    double fx, fy; to_ego((tx + 0.5) * TILE, (ty + 0.5) * TILE, fx, fy);
    if (fx < -1 || fx > L + 1 || std::fabs(fy) > hw + 1) continue;
    Tile* t = find(tx, ty); if (!t || !(t->flags & (T_OBST | T_NEG | T_LOWCLEAR | T_DIRTY))) continue;
    int k = cps(t->level); double cs = csize(t->level);
    for (int m = 0; m < ncell(t->level); m++) {
      const Cell& ce = t->cells[m]; if (!(ce.flags & TF_C_OBSERVED)) continue;
      int cxi, cyi; demorton(m << (2 * t->level), cxi, cyi); cxi >>= t->level; cyi >>= t->level; (void)k;
      to_ego(tx * TILE + (cxi + 0.5) * cs, ty * TILE + (cyi + 0.5) * cs, fx, fy);
      if (fx < 2.2 || fx > L || std::fabs(fy) > hw + cs * 0.5) continue;
      int w = 1 << (2 * t->level);                            // area weight in 5 cm cells
      if ((ce.occ >= 24 || (ce.occ && (ce.flags & TF_C_DYNAMIC))) && ce.top * 0.02 > 0.22) { int j = (ce.flags & TF_C_DYNAMIC) ? 1 : 0; cnt[j] += w; if (fx < dist[j]) { dist[j] = fx; kc[j] = cell_class(ce); val[j] = ce.top * 0.02; } }
      if (ce.neg >= 100) { cnt[2] += w; if (fx < dist[2]) { dist[2] = fx; val[2] = ce.neg / 255.0; } }
      if (ce.clear != 255 && ce.clear * 0.02 < cfg.veh_height + 0.2 && !(ce.occ >= 24 && ce.top)) { cnt[3] += w; if (fx < dist[3]) { dist[3] = fx; val[3] = ce.clear * 0.02; } }
    }
  }
  for (const Tile& t : tiles) if (t.used && (t.flags & T_NEG)) negc++;
  stats[TF_S_NEG_CELLS] = negc;
  for (int j = 0; j < 4; j++) if (cnt[j] >= 4 && dist[j] < best) { best = dist[j]; kind = j + 1; kcls = kc[j]; kval = val[j]; }
  // moving objects about to enter the corridor
  double ttc = 1e9; int tid = -1; int tcls = 0;
  for (const Track& k : tracks) { if (!k.dyn || k.miss > 2) continue;
    for (double q = 0; q <= 4.0; q += 0.25) { double fx, fy; to_ego(k.x + k.vx * q, k.y + k.vy * q, fx, fy);
      double egox = v * q; if (fx > 0 && fx < L && std::fabs(fy) < hw + 0.5 * k.ext && std::fabs(fx - egox) < 4.0 + 0.5 * k.ext) { if (q < ttc) { ttc = q; tid = k.id; tcls = k.votes[1] > k.votes[0] ? TF_PERSON : TF_VEHICLE; } break; } } }
  static const char* CN[8] = {"object", "ground", "terrain", "structure", "pole", "vegetation", "vehicle", "person"};
  double stopd = v * v / (2 * 3.0) + 4.0; const char* state = "CLEAR"; char reason[200] = "corridor clear";
  if (kind) {
    state = best < stopd ? "STOP" : "CAUTION";
    if (kind == 1) std::snprintf(reason, sizeof reason, "%s %.1f m ahead in path (%.1f m tall)", CN[kcls], best, kval);
    if (kind == 2) std::snprintf(reason, sizeof reason, "moving %s in path, %.1f m ahead", CN[kcls], best);
    if (kind == 3) std::snprintf(reason, sizeof reason, "negative obstacle %.1f m ahead (confidence %.2f)", best, kval);
    if (kind == 4) std::snprintf(reason, sizeof reason, "overhead clearance %.2f m, need %.1f m, %.1f m ahead", kval, cfg.veh_height + 0.2, best);
  }
  if (tid >= 0 && (!kind || std::strcmp(state, "STOP"))) {
    state = ttc < 2.0 ? "STOP" : "CAUTION";
    std::snprintf(reason, sizeof reason, "%s #%d crossing path, time to conflict %.1f s", CN[tcls], tid, ttc);
  }
  char buf[512];
  std::snprintf(buf, sizeof buf, "{\"state\":\"%s\",\"reason\":\"%s\",\"kind\":%d,\"dist\":%.2f,\"corridor_len\":%.1f,\"corridor_hw\":%.2f,\"speed\":%.2f,\"yaw\":%.4f,\"x\":%.3f,\"y\":%.3f,\"z\":%.3f}",
                state, reason, kind, kind ? best : -1.0, L, hw, v, eyaw, ex, ey, ez);
  decision = buf; (void)c; (void)s;
}

void TFEngine::reset_frame() {
  for (int lt : ttouched) { tcnt[lt] = 0; tgsum[lt] = 0; tgn[lt] = 0; ts1[lt] = 0; ts2[lt] = 0; tobst[lt] = 0; tai[lt] = 0; tlow[lt] = 0; trisk[lt] = 0; trough[lt] = -1; tacc[lt] = 0; }
  ttouched.clear();
  for (const Pillar& p : pil) pslot[p.lidx] = -1;
  pil.clear();
}

// =============================================================================== C API
extern "C" {

const char* tf_version(void) { return "terra-fovea 1.0"; }

TFEngine* tf_create(const char* cfg) {
  TFEngine* e = new TFEngine();
  std::string s = cfg ? cfg : "";
  size_t p = 0;
  while (p < s.size()) {
    size_t q = s.find(';', p); if (q == std::string::npos) q = s.size();
    std::string kv = s.substr(p, q - p); size_t eq = kv.find('=');
    if (eq != std::string::npos) {
      std::string k = kv.substr(0, eq); double v = std::atof(kv.c_str() + eq + 1); Cfg& c = e->cfg;
      if (k == "policy") c.policy = (int)v; else if (k == "uniform_level") c.uniform_level = (int)v;
      else if (k == "selective_ai") c.selective_ai = (int)v; else if (k == "ray_clear") c.ray_clear = (int)v;
      else if (k == "use_net") c.use_net = (int)v; else if (k == "threads") c.threads = (int)v;
      else if (k == "r0") c.r0 = v; else if (k == "r1") c.r1 = v; else if (k == "r2") c.r2 = v; else if (k == "r3") c.r3 = v;
      else if (k == "max_range") c.max_range = v; else if (k == "veh_height") c.veh_height = v;
      else if (k == "veh_width") c.veh_width = v; else if (k == "rough_thr") c.rough_thr = v; else if (k == "vres") c.vres = v;
    }
    p = q + 1;
  }
  e->cfg.max_range = std::min(e->cfg.max_range, 100.0); e->cfg.r3 = std::min(e->cfg.r3, e->cfg.max_range);
  e->cfg.uniform_level = std::min(LMAX, std::max(0, e->cfg.uniform_level));
#ifdef _OPENMP
  if (e->cfg.threads > 0) omp_set_num_threads(e->cfg.threads);
#endif
  return e;
}
void tf_destroy(TFEngine* e) { delete e; }

int tf_load_model(TFEngine* e, const char* path) {
  FILE* f = std::fopen(path, "rb"); if (!f) return -1;
  char magic[8]; int32_t d[5]; Net& N = e->net; N.ok = false;
  if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "TFNET01\0", 8) || std::fread(d, 4, 5, f) != 5) { std::fclose(f); return -2; }
  N.nf = d[0]; N.h1 = d[1]; N.h2 = d[2]; N.h3 = d[3]; N.nc = d[4];
  if (N.nf != TF_NFEAT || N.h1 > 64 || N.h2 > 64 || N.h3 > 128 || N.nc != TF_NCLS) { std::fclose(f); return -3; }
  auto rd = [&](std::vector<float>& v, size_t k) { v.resize(k); return std::fread(v.data(), 4, k, f) == k; };
  bool ok = rd(N.mu, N.nf) && rd(N.sd, N.nf) && rd(N.W1, (size_t)N.h1 * N.nf) && rd(N.b1, N.h1) && rd(N.W2, (size_t)N.h2 * N.h1) && rd(N.b2, N.h2)
         && rd(N.W3, (size_t)N.h3 * 3 * N.h2) && rd(N.b3, N.h3) && rd(N.W4, (size_t)N.nc * N.h3) && rd(N.b4, N.nc);
  std::fclose(f); N.ok = ok; return ok ? 0 : -4;
}

int tf_process(TFEngine* e, const float* pts, int n, const double* P, double t) {
  double t0 = now_ms(), a = t0, b; double* S = e->stats;
  e->frame++; e->n = n; e->reset_frame();
  double dt = (e->last_t >= 0 && t > e->last_t) ? t - e->last_t : 0.1;
  if (e->have_pose) { double vx = (P[3] - e->ex) / dt, vy = (P[7] - e->ey) / dt; e->evx = 0.6 * e->evx + 0.4 * vx; e->evy = 0.6 * e->evy + 0.4 * vy; }
  e->have_pose = true; e->last_t = t;
  e->stage_ground(pts, P);            b = now_ms(); S[TF_S_MS_GROUND] = b - a; a = b;
  e->stage_triage(nullptr, nullptr, nullptr, nullptr); b = now_ms(); S[TF_S_MS_TRIAGE] = b - a; a = b;
  e->stage_net();                     b = now_ms(); S[TF_S_MS_NET] = b - a; a = b;
  e->stage_track(dt);                 b = now_ms(); S[TF_S_MS_TRACK] = b - a; a = b;
  e->stage_risk();                    b = now_ms(); S[TF_S_MS_RISK] = b - a; a = b;
  e->stage_insert();                  b = now_ms(); S[TF_S_MS_INSERT] = b - a; a = b;
  S[TF_S_CLEARED] = 0;
  e->stage_rayclear();                b = now_ms(); S[TF_S_MS_RAYCLEAR] = b - a; a = b;
  e->stage_decision();                b = now_ms(); S[TF_S_MS_DECISION] = b - a; S[TF_S_MS_TOTAL] = b - t0;
  long ai = 0, np = 0; for (int i = 0; i < n; i++) if (e->ptile[i] >= 0) { np++; if (e->pflag[i] & TF_P_AI) ai++; }
  long nt = 0, ca = 0, co = 0, lv[5] = {0, 0, 0, 0, 0};
  for (const Tile& tl : e->tiles) if (tl.used) { nt++; int k = ncell(tl.level); ca += k; lv[tl.level] += k;
    for (int i = 0; i < k; i++) if (tl.cells[i].flags & TF_C_OBSERVED) co++; }
  S[TF_S_POINTS] = (double)np; S[TF_S_AI_POINTS] = (e->net.ok && e->cfg.use_net) ? (double)ai : 0; S[TF_S_TILES] = (double)nt;
  S[TF_S_CELLS_ALLOC] = (double)ca; S[TF_S_CELLS_OBSERVED] = (double)co;
  S[TF_S_MAP_BYTES] = (double)(ca * sizeof(Cell) + nt * sizeof(Tile) + e->dir.size() * sizeof(int32_t));
  for (int k = 0; k < 5; k++) S[TF_S_LEVEL0 + k] = (double)lv[k];
  int nd = 0; for (const Track& k : e->tracks) if (k.dyn) nd++;
  S[TF_S_TRACKS] = (double)e->tracks.size(); S[TF_S_DYNAMIC_TRACKS] = nd; S[TF_S_SPLITS] = (double)e->splits; S[TF_S_MERGES] = (double)e->merges;
  S[TF_S_FRAME] = e->frame;
  S[TF_S_WORKSPACE_BYTES] = (double)(e->pslot.size() * 4 + e->g1slot.size() * 4 + e->pil.capacity() * sizeof(Pillar) + (size_t)WT * WT * 60 + (size_t)n * 30);
  return 0;
}

void tf_get_stats(const TFEngine* e, double* out, int n) { for (int i = 0; i < n && i < TF_S_COUNT; i++) out[i] = e->stats[i]; }

int tf_point_labels(const TFEngine* e, uint8_t* cls, uint8_t* flags, uint8_t* conf, int n) {
  int m = std::min(n, e->n);
  for (int i = 0; i < m; i++) { if (cls) cls[i] = e->pcls[i]; if (flags) flags[i] = e->pflag[i]; if (conf) conf[i] = e->pconf[i]; }
  return m;
}

int tf_count_cells(const TFEngine* e) {
  int c = 0; for (const Tile& t : e->tiles) if (t.used) for (int i = 0; i < ncell(t.level); i++) if (t.cells[i].flags & TF_C_OBSERVED) c++;
  return c;
}

int tf_export_cells(const TFEngine* e, float* out, int cap) {
  int c = 0;
  for (const Tile& t : e->tiles) { if (!t.used) continue; double cs = csize(t.level);
    for (int m = 0; m < ncell(t.level); m++) { const Cell& ce = t.cells[m]; if (!(ce.flags & TF_C_OBSERVED)) continue;
      if (c >= cap) return c; int cx, cy; demorton(m << (2 * t.level), cx, cy); cx >>= t.level; cy >>= t.level;
      float* o = out + (size_t)TF_CELL_FLOATS * c++; int conf; int cl = cell_class(ce, &conf);
      o[0] = (float)(t.tx * TILE + (cx + 0.5) * cs); o[1] = (float)(t.ty * TILE + (cy + 0.5) * cs); o[2] = (float)cs;
      o[3] = ce.g * 0.01f; o[4] = has_obstacle(ce) ? ce.top * 0.02f : 0.f; o[5] = ce.clear == 255 ? -1.f : ce.clear * 0.02f;
      o[6] = (float)cl; o[7] = conf / 255.f; o[8] = 1.f - std::exp(-ce.occ / 32.f); o[9] = (float)ce.flags; o[10] = ce.neg / 255.f;
      o[11] = (float)t.level; o[12] = t.risk / 255.f; } }
  return c;
}

int tf_export_tracks(const TFEngine* e, float* out, int cap) {
  int c = 0;
  for (const Track& k : e->tracks) { if (c >= cap) break; if (k.miss > 2 || k.hits < 2) continue; float* o = out + (size_t)TF_TRACK_FLOATS * c++;
    o[0] = (float)k.id; o[1] = (float)k.x; o[2] = (float)k.y; o[3] = (float)k.vx; o[4] = (float)k.vy; o[5] = k.ext;
    o[6] = k.votes[1] > k.votes[0] ? (float)TF_PERSON : (float)TF_VEHICLE; o[7] = k.dyn ? 1.f : 0.f; o[8] = (float)k.hits; o[9] = k.path ? 1.f : 0.f; }
  return c;
}

const char* tf_decision_json(const TFEngine* e) { return e->decision.c_str(); }

// delta stream: [u32 ntiles] then per tile [i32 tx][i32 ty][u8 level][u8 risk][u16 n] + n * 8 bytes
// cell record: [u8 cx][u8 cy][i16 ground_cm][u8 top_2cm][u8 clear_2cm][u8 class|flags<<3][u8 conf<<4|occ]
int tf_export_delta(TFEngine* e, uint8_t* out, int cap, int full) {
  std::vector<uint8_t> buf; buf.resize(4); uint32_t nt = 0; std::vector<uint8_t> rec;
  auto put = [&](const void* p, size_t k) { const uint8_t* b = (const uint8_t*)p; buf.insert(buf.end(), b, b + k); };
  if (full) e->removed.clear();
  for (auto& r : e->removed) { int32_t a = r.first, b = r.second; uint8_t z[2] = {0, 0}; uint16_t k = 0xFFFF; put(&a, 4); put(&b, 4); put(z, 2); put(&k, 2); nt++; }
  e->removed.clear();
  double climit = (e->cfg.veh_height + 0.2) * 50.0;
  for (Tile& t : e->tiles) { if (!t.used) continue; rec.clear(); uint16_t k = 0;
    for (int m = 0; m < ncell(t.level); m++) { const Cell& ce = t.cells[m]; if (!(ce.flags & TF_C_OBSERVED)) continue;
      int cx, cy; demorton(m << (2 * t.level), cx, cy); cx >>= t.level; cy >>= t.level; int conf; int cl = cell_class(ce, &conf);
      uint8_t fl = (uint8_t)cl; if (ce.flags & TF_C_DYNAMIC) fl |= 8; if (ce.neg >= 100) fl |= 16; if (ce.flags & TF_C_INHERITED) fl |= 32;
      if (ce.clear != 255 && ce.clear < climit) fl |= 64;
      uint8_t r[8] = {(uint8_t)cx, (uint8_t)cy, 0, 0, (uint8_t)(has_obstacle(ce) ? ce.top : 0), ce.clear, fl,
                      (uint8_t)(((conf >> 4) << 4) | std::min(15, ce.occ / 8))};
      std::memcpy(r + 2, &ce.g, 2); rec.insert(rec.end(), r, r + 8); k++; }
    uint32_t h = 2166136261u ^ t.level; h = (h * 16777619u) ^ (t.risk >> 4);
    // change detection ignores sub-3 cm ground jitter and confidence wobble, so a settled tile is not resent
    for (size_t q = 0; q < rec.size(); q += 8) { int16_t g; std::memcpy(&g, &rec[q + 2], 2);
      uint8_t st[6] = {rec[q], rec[q + 1], (uint8_t)((g + 30000) / 3), (uint8_t)(rec[q + 4] >> 2), (uint8_t)(rec[q + 5] >> 2), rec[q + 6]};
      for (uint8_t b : st) h = (h ^ b) * 16777619u; }
    if (!h) h = 1;
    if (!full && h == t.exp_hash) continue;
    t.exp_hash = h; put(&t.tx, 4); put(&t.ty, 4); put(&t.level, 1); put(&t.risk, 1); put(&k, 2); put(rec.data(), rec.size()); nt++; }
  std::memcpy(buf.data(), &nt, 4);
  if ((int)buf.size() <= cap) std::memcpy(out, buf.data(), buf.size());
  else for (Tile& t : e->tiles) t.exp_hash = 0;                // caller must retry with a bigger buffer + full
  return (int)buf.size();
}

int tf_features(TFEngine* e, const float* pts, int n, const double* P, float* feats, int32_t* g1, int32_t* g2, uint8_t* tri) {
  e->n = n; e->reset_frame(); e->stage_ground(pts, P); e->stage_triage(feats, g1, g2, tri);
  return 0;
}

// ------------------------------------------------------------------------------- self test
int tf_selftest(int verbose) {
  int fail = 0, total = 0;
  auto check = [&](bool ok, const char* name) { total++; if (!ok) fail++; if (verbose || !ok) std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name); };
  { bool ok = true; for (int y = 0; y < 16; y++) for (int x = 0; x < 16; x++) { int cx, cy; demorton(morton(x, y), cx, cy); ok &= (cx == x && cy == y); }
    check(ok, "Z-order encode/decode is a bijection on 16x16"); }
  { bool ok = true; for (int l = 0; l < LMAX; l++) for (int y = 0; y < cps(l); y++) for (int x = 0; x < cps(l); x++) {
      int child = morton(x << l, y << l) >> (2 * l), parent = morton((x / 2) << (l + 1), (y / 2) << (l + 1)) >> (2 * (l + 1)); ok &= (child >> 2) == parent; }
    check(ok, "Z-order: parent index == child index >> 2 at every level"); }
  { bool ok = true; for (int l = 0; l <= LMAX; l++) ok &= std::fabs(ncell(l) * csize(l) * csize(l) - TILE * TILE) < 1e-12 && std::fabs(cps(l) * csize(l) - TILE) < 1e-12;
    check(ok, "every level tiles a 0.8 m tile exactly (no gap, no overlap at resolution borders)"); }
  std::srand(7);
  { bool ok = true; for (int it = 0; it < 2000; it++) { Cell p; std::memset(&p, 0, sizeof p); p.flags = TF_C_OBSERVED; p.occ = (uint8_t)(std::rand() % 256); p.top = 40; p.clear = 255;
      Cell ch[4]; split4(p, ch); int s = 0; for (auto& c : ch) s += c.occ; Cell m = merge4(ch); ok &= (s == p.occ) && (m.occ == p.occ); }
    check(ok, "occupancy evidence is conserved exactly by split and by merge(split(x))"); }
  { bool ok = true; for (int it = 0; it < 2000; it++) { Cell ch[4]; int topabs = -100000, clr = 100000; bool neg = false, dyn = false; int occ = 0;
      for (auto& c : ch) { std::memset(&c, 0, sizeof c); c.clear = 255; c.flags = TF_C_OBSERVED | TF_C_GROUND; c.g = (int16_t)(std::rand() % 60 - 30); c.gcnt = 1;
        if (std::rand() % 3 == 0) { c.top = (uint8_t)(1 + std::rand() % 100); c.occ = (uint8_t)(1 + std::rand() % 60); topabs = std::max(topabs, c.g + 2 * c.top); occ += c.occ; }
        if (std::rand() % 5 == 0) { c.clear = (uint8_t)(60 + std::rand() % 100); clr = std::min(clr, c.g + 2 * c.clear); }
        if (std::rand() % 7 == 0) { c.neg = 200; c.flags |= TF_C_NEGATIVE; neg = true; } if (std::rand() % 7 == 0) { c.flags |= TF_C_DYNAMIC; dyn = true; } }
      Cell m = merge4(ch);
      if (topabs > -100000) ok &= (m.g + 2 * m.top >= topabs) && m.occ >= std::min(255, occ);
      if (clr < 100000) ok &= (m.clear != 255) && (m.g + 2 * m.clear <= clr);
      ok &= (!neg || ((m.flags & TF_C_NEGATIVE) && m.neg == 200)) && (!dyn || (m.flags & TF_C_DYNAMIC)); }
    check(ok, "merge never lowers an obstacle top, never raises a clearance, never drops a hazard flag"); }
  { Cell ch[4]; for (int i = 0; i < 4; i++) { std::memset(&ch[i], 0, sizeof(Cell)); ch[i].clear = 255; ch[i].flags = TF_C_OBSERVED | TF_C_GROUND; ch[i].gcnt = 1; ch[i].g = (int16_t)(i < 2 ? 0 : 15); }
    Cell m = merge4(ch); check(m.grange >= 15, "a 15 cm kerb inside a coarse cell survives as ground-spread"); }
  { TFEngine e; Tile* t = e.create(3, -5, 0); bool ok = true;
    int a = morton(5, 9); t->cells[a].flags = TF_C_OBSERVED | TF_C_GROUND | TF_C_NEGATIVE; t->cells[a].neg = 180; t->cells[a].g = -40; t->cells[a].gcnt = 1;
    int b = morton(12, 2); t->cells[b].flags = TF_C_OBSERVED; t->cells[b].top = 85; t->cells[b].occ = 96; t->cells[b].g = 0;
    resample(*t, 4); ok &= t->level == 4 && (t->cells[0].flags & TF_C_NEGATIVE) && t->cells[0].occ == 96 && t->cells[0].g + 2 * t->cells[0].top >= 170;
    resample(*t, 0); int s = 0, nneg = 0; for (int i = 0; i < 256; i++) { s += t->cells[i].occ; if (t->cells[i].flags & TF_C_NEGATIVE) nneg++; }
    ok &= (s == 96) && nneg == 256 && t->level == 0;
    check(ok, "tile resample 5cm -> 80cm -> 5cm keeps hazards and total occupancy evidence"); }
  { TFEngine e; e.cfg.policy = 1; bool ok = true; int n = 20000; std::vector<float> pts((size_t)n * 4); double P[16] = {1, 0, 0, -3.3, 0, 1, 0, 7.7, 0, 0, 1, 1.8, 0, 0, 0, 1};
    for (int i = 0; i < n; i++) { double a = 6.2831853 * std::rand() / RAND_MAX, r = 3 + 95.0 * std::rand() / RAND_MAX; pts[4 * i] = (float)(r * std::cos(a)); pts[4 * i + 1] = (float)(r * std::sin(a)); pts[4 * i + 2] = -1.8f; pts[4 * i + 3] = 0.3f; }
    tf_process(&e, pts.data(), n, P, 0.0);
    int inside = 0, lvl_ok = 0, cnt = 0;
    for (int i = 0; i < n; i++) { if (e.ptile[i] < 0) continue; cnt++; double X = e.wx[i] + e.ex, Y = e.wy[i] + e.ey; int tx = floordiv(X, TILE), ty = floordiv(Y, TILE);
      Tile* t = e.find(tx, ty); if (!t) continue; double cs = csize(t->level); int cx = (int)((X - tx * TILE) / cs), cy = (int)((Y - ty * TILE) / cs);
      const Cell& c = t->cells[morton(cx << t->level, cy << t->level) >> (2 * t->level)];
      if ((c.flags & TF_C_OBSERVED) && X >= tx * TILE + cx * cs - 1e-9 && X < tx * TILE + (cx + 1) * cs + 1e-9) inside++;
      double d = std::hypot(X - e.ex, Y - e.ey); int want = d < 10 ? 0 : d < 20 ? 1 : d < 40 ? 2 : 3; if (t->level <= want) lvl_ok++; }
    ok &= inside == cnt && lvl_ok == cnt && cnt > n * 0.9;
    check(ok, "projection: every point lands in exactly one observed cell, at least as fine as its distance band"); }
  { TFEngine e; e.cfg.policy = 0; e.frame = 10; e.ex = 0; e.ey = 0; e.ez = 1.8; Tile* t = e.create(12, 0, 0); t->flags = T_OBST;
    for (int i = 0; i < 256; i++) { t->cells[i].flags = TF_C_OBSERVED; t->cells[i].top = 100; t->cells[i].occ = 60; t->cells[i].ostamp = 3; t->cells[i].age = 20; }
    e.n = 600; e.wx.assign(600, 30.f); e.wy.resize(600); e.wz.assign(600, 1.0f); e.rng.assign(600, 30.f); e.ptile.assign(600, 0);
    for (int i = 0; i < 600; i++) e.wy[i] = 2.4f * i / 600.f;
    for (int k = 0; k < 8; k++) e.stage_rayclear();
    int left = 0; for (int i = 0; i < 256; i++) if (t->cells[i].occ) left++;
    if (verbose) std::printf("    left=%d cleared=%ld\n", left, e.cleared);
    check(left < 26 && e.cleared > 200, "ray clearing removes a stale obstacle column that rays now pass through"); }
  if (verbose) std::printf("  %d/%d checks passed\n", total - fail, total);
  return fail;
}

} // extern "C"

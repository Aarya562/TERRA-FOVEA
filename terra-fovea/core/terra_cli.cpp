// terra_cli: run the engine on a SemanticKITTI-layout sequence with no Python in the loop.
//   terra_cli selftest
//   terra_cli <sequence_dir> [model.bin] ["policy=2;ray_clear=1"] [max_frames]
#include "terra_fovea.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static void mul(const double* A, const double* B, double* C) {
  for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) { double s = 0; for (int k = 0; k < 4; k++) s += A[i * 4 + k] * B[k * 4 + j]; C[i * 4 + j] = s; }
}
static void inv_rigid(const double* A, double* B) {
  for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) B[i * 4 + j] = A[j * 4 + i];
  for (int i = 0; i < 3; i++) B[i * 4 + 3] = -(B[i * 4] * A[3] + B[i * 4 + 1] * A[7] + B[i * 4 + 2] * A[11]);
  B[12] = B[13] = B[14] = 0; B[15] = 1;
}

int main(int argc, char** argv) {
  if (argc < 2) { std::printf("usage: terra_cli selftest | <sequence_dir> [model.bin] [cfg] [max_frames]\n"); return 1; }
  if (!std::strcmp(argv[1], "selftest")) { int f = tf_selftest(1); std::printf(f ? "SELFTEST FAILED (%d)\n" : "SELFTEST OK\n", f); return f; }
  std::string dir = argv[1];
  TFEngine* e = tf_create(argc > 3 ? argv[3] : "policy=2");
  if (argc > 2 && std::strlen(argv[2]) && tf_load_model(e, argv[2])) std::printf("warning: could not load model %s (geometry-only mode)\n", argv[2]);
  int maxf = argc > 4 ? std::atoi(argv[4]) : 1000000;
  double Tr[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, Tri[16];
  { std::ifstream c(dir + "/calib.txt"); std::string line;
    while (std::getline(c, line)) if (!line.compare(0, 3, "Tr:")) { std::istringstream ss(line.substr(3)); for (int i = 0; i < 12; i++) ss >> Tr[i]; } }
  inv_rigid(Tr, Tri);
  std::ifstream poses(dir + "/poses.txt"); std::string line; int f = 0; double sum = 0, mx = 0, S[TF_S_COUNT];
  while (f < maxf && std::getline(poses, line)) {
    double P[16] = {0}, A[16], B[16]; P[15] = 1; std::istringstream ss(line); for (int i = 0; i < 12; i++) ss >> P[i];
    mul(Tri, P, A); mul(A, Tr, B);                       // camera-frame pose -> lidar-frame pose
    char name[64]; std::snprintf(name, sizeof name, "/velodyne/%06d.bin", f);
    std::ifstream bin(dir + name, std::ios::binary | std::ios::ate); if (!bin) break;
    size_t bytes = (size_t)bin.tellg(); bin.seekg(0); std::vector<float> pts(bytes / 4); bin.read((char*)pts.data(), bytes);
    tf_process(e, pts.data(), (int)(pts.size() / 4), B, f * 0.1);
    tf_get_stats(e, S, TF_S_COUNT); if (f >= 5) { sum += S[TF_S_MS_TOTAL]; if (S[TF_S_MS_TOTAL] > mx) mx = S[TF_S_MS_TOTAL]; }
    if (f % 20 == 0) std::printf("frame %4d  %6.0f pts  %5.1f ms  cells %7.0f  map %6.2f MB  tracks %2.0f (%1.0f moving)  %s\n", f, S[TF_S_POINTS],
                                 S[TF_S_MS_TOTAL], S[TF_S_CELLS_ALLOC], S[TF_S_MAP_BYTES] / 1048576.0, S[TF_S_TRACKS], S[TF_S_DYNAMIC_TRACKS], tf_decision_json(e));
    f++;
  }
  if (f > 5) std::printf("\n%d frames  mean %.1f ms (%.1f FPS)  worst %.1f ms\n", f, sum / (f - 5), 1000.0 * (f - 5) / sum, mx);
  tf_destroy(e); return 0;
}

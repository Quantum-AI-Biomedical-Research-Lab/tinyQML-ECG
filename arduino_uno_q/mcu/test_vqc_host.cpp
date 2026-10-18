#include <chrono>
#include <cmath>
#include <cstdio>

#include "test_vectors.h"
#include "vqc.h"
#include "vqc_params.h"

int main() {
  double maxErr = 0.0;
  const auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < VQC_N_TESTS; i++) {
    const float p = vqc_predict_features(VQC_TEST_X[i]);
    const double e = std::fabs((double)p - (double)VQC_TEST_P[i]);
    if (e > maxErr) maxErr = e;
    std::printf("%2d  p_c=%.6f  p_numpy=%.6f  |dp|=%.2e\n", i, p, VQC_TEST_P[i], e);
  }
  const auto t1 = std::chrono::high_resolution_clock::now();
  const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / VQC_N_TESTS;
  std::printf("\nmax |dp| = %.3e, mean %.2f us per inference on host\n", maxErr, us);
  std::printf("state %zu B, parameters %zu B\n", sizeof(float) * 2 * (1 << VQC_N_QUBITS),
              sizeof(VQC_ENC_W) + sizeof(VQC_ENC_B) + sizeof(VQC_ROT) + sizeof(VQC_HEAD_W) + sizeof(float));
  return maxErr < 1e-5 ? 0 : 1;
}
#include <stdlib.h>
#include <math.h>
#include "base/spmv.h"
#include <dmlc/io.h>
#include <dmlc/timer.h>
#include "data/row_block.h"

float norm(const std::vector<float>& v) {
  float r = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    r += v[i] * v[i];
  }
  return sqrt(r);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: <libsvm>\n");
    return 0;
  }
  using namespace dmlc;
  RowBlockIter<unsigned> *iter = RowBlockIter<unsigned>::Create(
      argv[1], 0, 1, "libsvm");

  data::RowBlockContainer<unsigned> data;

  double tv = GetTime();
  while (iter->Next()) {
    data.Push(iter->Value());
  }
  size_t nrow = data.offset.size() - 1;
  size_t ncol = data.max_index + 1;
  LOG(INFO) << "read a " << nrow << " x "
            << ncol << " matrix in " << GetTime() - tv << " sec";

  RowBlock<unsigned> D = data.GetBlock();
  std::vector<int> nthreads = {1, 2, 4, 8, 16};

  // Times
  std::vector<float> x(ncol);
  for (size_t i = 0; i < ncol; ++i) {
    x[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  std::vector<float> y(nrow);

  // warmup
  SpMV::Times(D, x, &y, 1);
  float base_v = norm(y);
  float base_t = 0;

  for (size_t i = 0; i < nthreads.size(); ++i) {
    double tv = GetTime();
    SpMV::Times(D, x, &y, nthreads[i]);
    double t = GetTime() - tv;

    if (i == 0) base_t = t;
    LOG(INFO) << "Times: " << nthreads[i] << " threads, "
              << t << " sec, "
              << base_t / t << " speedup";
    float ret = norm(y);
    CHECK_LT(fabs(ret-base_v), 1e-6);
  }

  // Trans Times
  x.resize(nrow);
  for (size_t i = 0; i < nrow; ++i) {
    x[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
  }
  y.resize(ncol);

  // warmup
  SpMV::TransTimes(D, x, &y, 1);
  base_v = norm(y);
  base_t = 0;

  for (size_t i = 0; i < nthreads.size(); ++i) {
    double tv = GetTime();
    SpMV::TransTimes(D, x, &y, nthreads[i]);
    double t = GetTime() - tv;
    if (i == 0) base_t = t;
    LOG(INFO) << "TransTimes: " << nthreads[i] << " threads, "
              << t << " sec, "
              << base_t / t << " speedup";
    float ret = norm(y);
    CHECK_LT(fabs(ret-base_v), 1e-6);
  }

  return 0;
}

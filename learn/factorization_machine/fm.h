#pragma once

// #include "base/dist_monitor.h"
#include "base/minibatch_iter.h"
#include "solver/async_sgd.h"
#include "config.pb.h"

namespace dmlc {
namespace fm {

using FeaID = ps::Key;
using Real = float;
template <typename T> using Blob = ps::Blob<T>;

// commands
static const int kPushFeaCnt = 1;

class Progress : public VectorProgress {
 public:
  Progress() : VectorProgress(6, 3) {}
  virtual ~Progress() { }

  virtual void Merge(const IProgress* const other) {

  }

  /// head string for printing
  virtual std::string HeadStr() {
    return "  objv       AUC    accuracy";
  }

  /// string for printing
  virtual std::string PrintStr(const IProgress* const prev) {
    return "";
  }


  // mutator
  double& objv() { return fvec_[0]; }
  double& acc() { return fvec_[1]; }
  double& auc() { return fvec_[2]; }
  double& logloss() { return fvec_[3]; }
  double& weight2() { return fvec_[4]; }
  double& wdelta2() { return fvec_[5]; }

  size_t& count() { return ivec_[0]; }
  size_t& num_ex() { return ivec_[1]; }
  size_t& nnz_w() { return ivec_[2]; }
};

class FMScheduler : public solver::AsyncSGDScheduler<Progress> {
 public:
  FMScheduler(const Config& conf) { }
  virtual ~FMScheduler() { }
};

}  // namespace fm
}  // namespace dmlc

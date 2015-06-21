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
  Progress() : VectorProgress(4,4) {}
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
};

class FMScheduler : public solver::AsyncSGDScheduler<Progress> {
 public:
  FMScheduler(const Config& conf) { }
  virtual ~FMScheduler() { }
};

}  // namespace fm
}  // namespace dmlc

#pragma once

#include "base/dist_monitor.h"
#include "base/minibatch_iter.h"
namespace dmlc {
namespace fm {

class AsyncSGDWorker { // }: public ps::App {
 public:
  AsyncSGDWorker(const Config& conf) : conf_(conf), reporter_(conf_.disp_itv())
 {
 }

  void Process()
};
}  // namespace fm
}  // namespace dmlc

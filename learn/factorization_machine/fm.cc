#include "ps.h"
#include "fm.h"
#include "base/arg2proto.h"
#include "proto/config.pb.h"

namespace ps {
App* App::Create(int argc, char *argv[]) {

  ::dmlc::fm::Config conf;
  ::dmlc::Arg2Proto(argc, argv, &conf);

  if (IsWorkerNode()) {
    // return new AsyncSGDWorker(conf);
  } else if (IsServerNode()) {
    return new ::dmlc::fm::FMServer(conf);
  } else if (IsSchedulerNode()) {
    return new ::dmlc::fm::FMWorker(conf);
  }
  // LOG(ERROR) << conf.ShortDebugString();
  LOG(FATAL) << "unknown node";
  return NULL;
}
}  // namespace ps

int main(int argc, char *argv[]) {
  return ps::RunSystem(&argc, &argv);
}

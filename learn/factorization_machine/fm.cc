#include "ps.h"
#include "fm.h"
#include "fm_server.h"
#include "fm_worker.h"
#include "base/arg_parser.h"
#include "config.pb.h"

namespace ps {
App* App::Create(int argc, char *argv[]) {
  CHECK_GE(argc, 2) << "\nusage: " << argv[0] << " conf_file";
  ::dmlc::ArgParser parser;
  if (strcmp(argv[1], "none")) parser.ReadFile(argv[1]);
  parser.ReadArgs(argc-2, argv+2);
  ::dmlc::fm::Config conf; parser.ParseToProto(&conf);

  if (IsWorkerNode()) {
    return new ::dmlc::fm::FMWorker(conf);
  } else if (IsServerNode()) {
    return new ::dmlc::fm::FMServer(conf);
  } else if (IsSchedulerNode()) {
    return new ::dmlc::fm::FMScheduler(conf);
  } else {
    LOG(FATAL) << "unknown node";
  }
  return NULL;
}
}  // namespace ps

int64_t dmlc::fm::ISGDHandle::new_w = 0;
int64_t dmlc::fm::ISGDHandle::new_V = 0;

int main(int argc, char *argv[]) {
  return ps::RunSystem(&argc, &argv);
}

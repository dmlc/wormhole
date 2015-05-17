#include "ps.h"
#include "sgd/async_sgd.h"

namespace ps {
App* App::Create(int argc, char *argv[]) {
  using namespace dmlc;
  using namespace dmlc::linear;
  CHECK_GE(argc, 2);
  ArgParser parser;
  parser.ReadFile(argv[1]);
  parser.ReadArgs(argc-2, argv+2);
  linear::Config conf; parser.ParseToProto(&conf);

  if (IsWorkerNode()) {
    return new AsyncSGDWorker(conf);
  } else if (IsServerNode()) {
    return new AsyncSGDServer(conf);
  } else if (IsSchedulerNode()){
    return new AsyncSGDScheduler(conf);
  }
  LOG(FATAL) << "unknown node";
  return NULL;
}
}  // namespace ps

int main(int argc, char *argv[]) {
  return ps::RunSystem(&argc, &argv);
}

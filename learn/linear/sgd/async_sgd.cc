#include "ps.h"
#include "sgd/async_sgd.h"

DEFINE_string(conf, "", "config file");
namespace ps {
App* App::Create(int argc, char *argv[]) {
  using namespace dmlc;
  using namespace dmlc::linear;

  ArgParser parser;
  if (!FLAGS_conf.empty()) parser.ReadFile(FLAGS_conf.c_str());
  parser.ReadArgs(argc-1, argv+1);
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

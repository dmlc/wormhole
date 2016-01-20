#include "async_sgd.h"
#include "config.pb.h"
#include "base/arg_parser.h"

namespace ps {
App* App::Create(int argc, char *argv[]) {
  CHECK_GE(argc, 2) << "\nusage: " << argv[0] << " conf_file";
  ::dmlc::ArgParser parser;
  if (strcmp(argv[1], "none")) parser.ReadFile(argv[1]);
  parser.ReadArgs(argc-2, argv+2);
  ::dmlc::svdfeature::Config conf; parser.ParseToProto(&conf);
  
  CHECK_EQ(conf.data_format(),"svdfeature") << "unsupported data format";

  NodeInfo n;
  if (n.IsWorker()) {
    return new ::dmlc::svdfeature::AsyncWorker(conf);
  } else if (n.IsServer()) {
    return new ::dmlc::svdfeature::AsyncServer(conf);
  } else if (n.IsScheduler()) {
    return new ::dmlc::svdfeature::AsyncScheduler(conf);
  } else {
    LOG(FATAL) << "unknown node";
  }
  return NULL;
}
}  // namespace ps

int64_t dmlc::svdfeature::ISGDHandle::new_w = 0;
int64_t dmlc::svdfeature::ISGDHandle::new_V = 0;

int main(int argc, char *argv[]) {
  return ps::RunSystem(&argc, &argv);
}

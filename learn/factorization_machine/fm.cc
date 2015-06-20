#include "ps.h"
#include "fm.h"
#include "base/arg2proto.h"
#include "proto/config.pb.h"
#include "dmlc/config.h"

namespace ps {
App* App::Create(int argc, char *argv[]) {

  ::dmlc::fm::Config conf;
  ::dmlc::Arg2Proto(argc, argv, &conf);

  LOG(ERROR) << conf.ShortDebugString();
  // LOG(FATAL) << "unknown node";
  return NULL;
}
}  // namespace ps

int main(int argc, char *argv[]) {
  // return ps::RunSystem(&argc, &argv);
ps::App::Create(argc, argv);
  return 0;
}

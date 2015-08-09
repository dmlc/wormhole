/**
 * @file   iter_solver_test.cc
 * @brief
 *
 * run: tracker/dmlc_local.py -s 2 -n 4 learn/test/build/iter_solver_test
 */

#include "solver/iter_solver.h"

DEFINE_string(train_data, "", "");
DEFINE_string(val_data, "", "");
DEFINE_bool(batch, false, "");

namespace dmlc {

class IterTestScheduler : public solver::IterScheduler {
 public:
  IterTestScheduler() {
    train_data_  = FLAGS_train_data;
    val_data_    = FLAGS_val_data;
    batch_       = FLAGS_batch;
    data_format_ = "libsvm";
  }
  virtual ~IterTestScheduler() { }

};

class IterTestServer : public solver::IterServer {
 public:
  IterTestServer() { }
  virtual ~IterTestServer() { }

  virtual void SaveModel(Stream* fo) const { fo->Write(model_); }
  virtual void LoadModel(Stream* fi) { fi->Read(&model_); }

 private:
  std::vector<float> model_;  // a fake model
};

class IterTestWorker : public solver::IterWorker {
 public:
  IterTestWorker() { }
  virtual ~IterTestWorker() { }

  virtual void Process(const Workload& wl) {
    printf("worker %d: %s\n", ps::NodeInfo::MyRank(), wl.ShortDebugString().c_str());
    srand(time(NULL));
    int t = rand() % 10000;
    usleep(t);
    std::vector<double> p(1, t);
    Report(p);
  }
};
}  // namespace dmlc

namespace ps {

App* App::Create(int argc, char *argv[]) {
  NodeInfo info;
  if (info.IsWorker()) {
    return new ::dmlc::IterTestWorker();
  } else if (info.IsServer()) {
    return new ::dmlc::IterTestServer();
  } else if (info.IsScheduler()) {
    return new ::dmlc::IterTestScheduler();
  }
  return NULL;
}

}  // namespace ps

int main(int argc, char *argv[]) {
  return ps::RunSystem(&argc, &argv);
}

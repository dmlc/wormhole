/**
 * @file   iter_solver_test.cc
 * @brief
 * on wormhole's root directory:
 \code
 make test
 seq 0 3 | xargs -I {} touch data/part-{}
 tracker/dmlc_local.py -s 2 -n 4 learn/test/build/iter_solver_test \
   -train_data data/part-[0-1] -val_data data/part-[2-3]
 \endcode
 * run:
 */

#include "solver/iter_solver.h"

DEFINE_string(train_data, "", "");
DEFINE_string(val_data, "", "");
DEFINE_bool(batch, false, "batch or online model");

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

  virtual std::string ProgHeader() const {
    return "   tic";
  }

  virtual std::string ProgString(const std::vector<double>& prog) const {
    if (prog.size()) return std::to_string(prog[0]);
    return "";
  }
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
    if (seedp_ == 0) seedp_ = ps::NodeInfo::MyRank();
    int t = (rand_r(&seedp_) % 100000) + 500000;
    usleep(t);
    std::vector<double> p(1, t);
    ReportToScheduler(p);
  }
 private:
  unsigned int seedp_ = 0;
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

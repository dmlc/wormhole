/**
 * @file   iter_solver_test.cc
 * @brief
 * on wormhole's root directory:
 \code
 make test
 seq 0 3 | xargs -I {} touch data/part-{}
 tracker/dmlc_local.py -s 2 -n 4 learn/test/build/iter_solver_test \
   -data data/ -model out
 \endcode
 * run:
 */

#include "solver/iter_solver.h"

DEFINE_string(data, "", "");
DEFINE_string(model, "", "");

namespace dmlc {

class IterTestScheduler : public solver::IterScheduler {
 public:
  IterTestScheduler() { }
  virtual ~IterTestScheduler() { }

  virtual bool Run() {
    data_filename_ = FLAGS_data;
    int max_iter = 3;
    for (int i = 0; i < max_iter; ++i) {
      std::cout << "iter = " << i << std::endl;
      OneIteration();
      Wait(SaveModel(FLAGS_model, i));
    }
    return true;
  }

  void OneIteration () {
    Start();
    while (!IsFinished()) {
      sleep(1);
      auto p = GetProgress();
      if (p.size())
        std::cout << "#workload : " << p[0] << " time: " << p[1]/1e6 << std::endl;
    }
  }

  virtual std::string ProgString(const std::vector<double>& prog) const {
    if (prog.size()) return std::to_string(prog[0]);
    return "";
  }
};

class IterTestServer : public solver::IterServer {
 public:
  IterTestServer() : model_(2) { }
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
    if (seedp_ == 0) seedp_ = ps::NodeInfo::MyRank();
    int t = (rand_r(&seedp_) % 100000) + 500000;
    usleep(t);
    std::vector<double> p = {1.0, (double)t};
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

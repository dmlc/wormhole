/**
 * @file   data_parallel_test.cc
 * @brief
 * on wormhole's root directory:
 \code
 make test
 seq 0 3 | xargs -I {} touch data/part-{}
 tracker/dmlc_local.py -s 2 -n 4 learn/test/build/data_parallel_test -data data/
 \endcode
 * one can also use `data/part-[0-1]` or `data/part.*`
 */

#include "solver/data_parallel.h"

DEFINE_string(data, "", "");
DEFINE_bool(batch, false, "batch or online model");

namespace dmlc {

class DataParTestScheduler : public solver::DataParScheduler {
 public:
  DataParTestScheduler() {
    data_filename_      = FLAGS_data;
    batch_              = FLAGS_batch;
    data_format_        = "libsvm";
    workload_.type      = Workload::TRAIN;
    workload_.data_pass = 8;
  }

  virtual ~DataParTestScheduler() { }

  virtual bool Run() {
    StartDispatch(); while (!IsFinished()) usleep(100000); return true;
  }
};

class DataParTestWorker : public solver::DataParWorker {
 public:
  DataParTestWorker() { }
  virtual ~DataParTestWorker() { }

  virtual void Process(const Workload& wl) {
    if (seedp_ == 0) seedp_ = ps::NodeInfo::MyRank();
    int t = (rand_r(&seedp_) % 100000) + 500000;
    usleep(t);
    printf("worker %d: %s, time=%d\n",
           ps::NodeInfo::MyRank(), wl.ShortDebugString().c_str(), t);
  }
 private:
  unsigned int seedp_ = 0;
};

}  // namespace dmlc

namespace ps {

App* App::Create(int argc, char *argv[]) {
  NodeInfo info;
  if (info.IsWorker()) {
    return new ::dmlc::DataParTestWorker();
  } else if (info.IsScheduler()) {
    return new ::dmlc::DataParTestScheduler();
  } else {
    return new App();
  }
}

}  // namespace ps

int main(int argc, char *argv[]) {
  return ps::RunSystem(&argc, &argv);
}

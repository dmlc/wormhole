// single machine version of online ftrl
#include <unordered_map>
#include <iostream>
#include <gflags/gflags.h>
#include "dmlc/timer.h"

#include "base/utils.h"
#include "base/minibatch_iter.h"
#include "base/arg_parser.h"
#include "base/localizer.h"
#include "base/loss.h"
#include "proto/config.pb.h"
#include "sgd/sgd_server_handle.h"
#include "base/workload_pool.h"
namespace dmlc {
namespace linear {

using FeaID = unsigned;

class LocalServer {
 public:
  LocalServer(const Config& conf) : conf_(conf) {
    if (conf_.has_lr_eta()) handle_.alpha = conf_.lr_eta();
    if (conf_.has_lr_beta()) handle_.beta = conf_.lr_beta();
    if (conf_.lambda_size() > 0) handle_.lambda1 = conf_.lambda(0);
    if (conf_.lambda_size() > 1) handle_.lambda2 = conf_.lambda(1);
    handle_.tracker = &tracker;
  }

  void Push(const std::vector<FeaID>& keys, const std::vector<real_t>& grad) {
    CHECK_EQ(keys.size(), grad.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      const FeaID* k = keys.data() + i;
      real_t* v = FindValue(*k);
      handle_.Push(Blob<const FeaID>(k, 1),
                   Blob<const real_t>(grad.data()+i, 1),
                   Blob<real_t>(v, kVS));
    }
  }

  void Pull(const std::vector<FeaID>& keys, std::vector<real_t>* weight) {
    CHECK_EQ(keys.size(), weight->size());
    for (size_t i = 0; i < keys.size(); ++i) {
      const FeaID* k = keys.data() + i;
      real_t* v = FindValue(*k);
      handle_.Pull(Blob<const FeaID>(k, 1),
                   Blob<const real_t>(v, kVS),
                   Blob<real_t>(weight->data()+i, 1));
    }
  }

  // size_t nnz() { return tracker.prog.ivec[1]; }

  const Progress& progress() { return tracker.prog; }
 private:
  static const int kVS = 3;  // value size
  real_t* FindValue(FeaID key) {
    auto it = data_.find(key);
    if (it == data_.end()) {
      // init if necessary
      real_t* v = data_[key];
      handle_.Init(Blob<const FeaID>(&key, 1),
                   Blob<real_t>(v, kVS));
      return v;
    }
    return it->second;
  }

  Config conf_;
  ModelMonitor tracker;
  FTRLHandle<FeaID, real_t> handle_;
  std::unordered_map<FeaID, real_t[kVS]> data_;
  template <typename T> using Blob = ps::Blob<T>;
};

class LocalWorker {
 public:
  LocalWorker(const Config& conf) : conf_(conf), server_(conf), num_ex_(0) { }

  void Run() {
    CHECK(conf_.has_train_data());
    printf("training...\n");
    WorkloadPool pool;
    for (int iter = 0; iter < conf_.max_data_pass(); ++iter) {
      pool.Add(conf_.train_data(), 1);
    }

    start_ = GetTime();
    Workload files;
    while (true) {
      pool.Get("", &files);
      if (!Process(files, conf_.disp_itv(), true)) break;
    }

    if (conf_.has_val_data()) {
      pool.Clear();
      mnt_.prog.Clear();
      num_ex_ = 0;
      printf("evaluation...\n");

      pool.Add(conf_.val_data(), 1);
      while (true) {
        pool.Get("", &files);
        if (!Process(files, 0, true)) break;
      }
      Print();
    }
  }

 private:
  void Print() {
    mnt_.prog.Merge(server_.progress());
    num_ex_ += mnt_.prog.num_ex();
    printf("%7.1lf sec, #ex %.3g, %s\n",
           GetTime() - start_, (double)num_ex_, mnt_.prog.PrintStr().c_str());
    mnt_.prog.Clear();
  }

  bool Process(const Workload& files, int disp, bool update) {
    if (files.file_size() == 0) return false;
    auto loss = CreateLoss<real_t>(conf_.loss());
    double tv = GetTime();
    File file = files.file(0);
    dmlc::data::MinibatchIter<FeaID> reader(
        file.file().c_str(), file.k(), file.n(),
        conf_.data_format().c_str(), conf_.minibatch());
    reader.BeforeFirst();
    while (reader.Next()) {
      // localize the minibatch
      auto global = reader.Value();
      dmlc::data::RowBlockContainer<unsigned> local;
      std::vector<FeaID> feaids;
      Localizer<FeaID> lc;
      lc.Localize(global, &local, &feaids);

      // fetch the weight
      std::vector<real_t> buf(feaids.size());
      server_.Pull(feaids, &buf);

      loss->Init(local.GetBlock(), buf);

      mnt_.Update(global.size, loss);

      if (disp > 0 && (GetTime() - tv > disp)) {
        Print();
        tv = GetTime();
      }

      if (update) {
        loss->CalcGrad(&buf);
        server_.Push(feaids, buf);
        // LOG(INFO) << "grad: " << DebugStr(buf);
      }
    }
    delete loss;
    return true;
  }

  Config conf_;
  LocalServer server_;
  WorkerMonitor mnt_;
  size_t num_ex_;
  double start_;
};

}  // namespace linear
}  // namespace dmlc

DEFINE_string(conf, "", "config file");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  using namespace dmlc;

  ArgParser parser;
  if (!FLAGS_conf.empty()) parser.ReadFile(FLAGS_conf.c_str());
  parser.ReadArgs(argc-1, argv+1);
  linear::Config conf; parser.ParseToProto(&conf);

  linear::LocalWorker worker(conf);
  worker.Run();

  return 0;
}

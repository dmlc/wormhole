// single machine version of online ftrl
#include <unordered_map>
#include <gflags/gflags.h>
#include "dmlc/timer.h"

#include "base/utils.h"
#include "base/minibatch_iter.h"
#include "base/arg_parser.h"
#include "base/localizer.h"
#include "base/loss.h"
#include "proto/config.pb.h"
#include "sgd/sgd_server_handle.h"
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
  LocalWorker(const Config& conf) : conf_(conf), server_(conf) { }

  void Run() {

    CHECK(conf_.has_train_data());
    dmlc::data::MinibatchIter<FeaID> reader(
        conf_.train_data().c_str(), 0, 1, conf_.data_format().c_str(),
        conf_.minibatch());

    for (int iter = 0; iter < conf_.max_data_pass(); ++iter) {
      Process(reader, 1, true);
      LOG(INFO) << "iter " << iter << " done";
    }

    if (conf_.has_val_data()) {
      dmlc::data::MinibatchIter<FeaID> val_reader(
          conf_.val_data().c_str(), 0, 1, conf_.data_format().c_str(),
          10000);
      mnt_.prog.Clear();
      Process(val_reader, 0, false);
    }
  }

 private:
  void Print() {
    mnt_.prog.Merge(server_.progress());
    auto& prog = mnt_.prog;
    size_t num_ex = mnt_.prog.num_ex();
    LOG(INFO) << "#ex " << num_ex
              << ", objv " << prog.objv() / prog.num_ex()
              << ", auc " << prog.auc() / prog.count()
              << ", acc " << prog.acc() / prog.count()
              << ", nnz w " << prog.nnz_w();
    mnt_.prog.Clear();
    mnt_.prog.num_ex() = num_ex;
  }

  void Process(
      dmlc::data::MinibatchIter<FeaID>& reader, int disp, bool update) {

    auto loss = CreateLoss<real_t>(conf_.loss());
    reader.BeforeFirst();
    double tv = GetTime();
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
    if (disp == 0) Print();
    delete loss;
  }

  Config conf_;
  LocalServer server_;
  WorkerMonitor mnt_;

};

}  // namespace linear
}  // namespace dmlc

DEFINE_string(conf, "", "config file");

int main(int argc, char *argv[]) {
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

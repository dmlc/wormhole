// single machine version of online ftrl
#include <unordered_map>
#include <gflags/gflags.h>

#include "base/utils.h"
#include "base/minibatch_iter.h"
#include "base/arg_parser.h"
#include "base/localizer.h"
#include "base/loss.h"
#include "proto/linear.pb.h"
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
      handle_.HandlePush(
          0, Blob<const FeaID>(k, 1), Blob<const real_t>(grad.data()+i, 1),
          Blob<real_t>(v, kVS));
    }
  }

  void Pull(const std::vector<FeaID>& keys, std::vector<real_t>* weight) {
    CHECK_EQ(keys.size(), weight->size());
    for (size_t i = 0; i < keys.size(); ++i) {
      const FeaID* k = keys.data() + i;
      real_t* v = FindValue(*k);
      handle_.HandlePull(
          0, Blob<const FeaID>(k, 1), Blob<const real_t>(v, kVS),
          Blob<real_t>(weight->data()+i, 1));
    }
  }

  size_t nnz() { return tracker.prog.ivec[1]; }
 private:
  static const int kVS = 3;  // value size
  real_t* FindValue(FeaID key) {
    auto it = data_.find(key);
    if (it == data_.end()) {
      // init if necessary
      real_t* v = data_[key];
      handle_.HandleInit(0, Blob<const FeaID>(&key, 1), Blob<real_t>(v, kVS));
      return v;
    }
    return it->second;
  }

  Config conf_;
  OnlineModelTracker tracker;
  FTRLHandle<FeaID, real_t> handle_;
  std::unordered_map<FeaID, real_t[kVS]> data_;
  template <typename T> using Blob = ps::Blob<T>;
};

class LocalWorker {
 public:
  LocalWorker(const Config& conf) : conf_(conf) { }
  void Run() {

    CHECK(conf_.has_train_data());
    dmlc::data::MinibatchIter<FeaID> reader(
        conf_.train_data().c_str(), 0, 1, conf_.data_format().c_str(),
        conf_.minibatch());

    LocalServer server(conf_);
    auto loss = CreateLoss<real_t>(conf_.loss());

    size_t num_ex = 0;
    int k = 0;
    for (int iter = 0; iter < conf_.max_data_pass(); ++iter) {
      reader.BeforeFirst();
      while (reader.Next()) {
        // localize the minibatch
        auto global = reader.Value();
        // LOG(INFO) << "global " << DebugStr(global);
        dmlc::data::RowBlockContainer<unsigned> local;
        std::vector<FeaID> feaids;
        Localizer<FeaID> lc;
        lc.Localize(global, &local, &feaids);

        // LOG(INFO) << "global " << DebugStr(local);
        // fetch the weight
        std::vector<real_t> buf(feaids.size());
        server.Pull(feaids, &buf);

        loss->Init(local.GetBlock(), buf);
        num_ex += global.size;

        if ( ++k % conf_.print_iter() == 0) {
          LOG(INFO) << "#ex " << num_ex
                    << ", objv " << loss->Objv() / global.size
                    << ", auc " << loss->AUC()
                    << ", acc " << loss->Accuracy()
                    << ", nnz " << server.nnz();
        // LOG(INFO) << "weight: " << DebugStr(buf);

        }

        loss->CalcGrad(&buf);
        server.Push(feaids, buf);

        // LOG(INFO) << "grad: " << DebugStr(buf);
      }
    }
  }

 private:
  Config conf_;
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

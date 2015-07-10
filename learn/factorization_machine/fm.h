#pragma once
#include "ps.h"
#include "solver/async_sgd.h"
#include "config.pb.h"
namespace dmlc {
namespace fm {

using FeaID = ps::Key;
using Real = float;
static const int kPushFeaCnt = 1;

class Progress : public VectorProgress {
 public:
  Progress() : VectorProgress(4, 4) {}
  virtual ~Progress() { }


  /// head string for printing
  virtual std::string HeadStr() {
    return " ttl #ex  inc #ex |  |w|_0  logloss_w |   |V|_0   logloss    AUC";
  }

  /// string for printing
  virtual std::string PrintStr(const IProgress* const prev) {
    Progress* const p = (Progress* const) prev;
    if (num_ex() == 0) return "";
    double cnt = (double)count();
    double num = (double)num_ex();
    char buf[256];
    snprintf(buf, 256, "%7.2g  %7.2g | %9.4g  %6.4lf | %9.4g  %6.4lf  %6.4lf ",
             (double)(p->num_ex() + num), num,
             (double)(p->nnz_w() + nnz_w()), objv_w() / num,
             (double)(p->nnz_V() + nnz_V()), objv() / num, auc() / cnt);
    return std::string(buf);
  }

  // mutator
  double& objv() { return fvec_[0]; }
  double objv() const { return fvec_[0]; }
  double& auc() { return fvec_[1]; }

  double& objv_w() { return fvec_[2]; }
  double& copc() { return fvec_[3]; }

  int64_t& count() { return ivec_[0]; }
  int64_t& num_ex() { return ivec_[1]; }
  int64_t num_ex() const { return ivec_[1]; }
  int64_t& nnz_w() { return ivec_[2]; }
  int64_t nnz_w() const { return ivec_[2]; }
  int64_t& nnz_V() { return ivec_[3]; }
  int64_t nnz_V() const { return ivec_[3]; }
};

class FMScheduler : public solver::AsyncSGDScheduler<Progress> {
 public:
  FMScheduler(const Config& conf) : conf_(conf) {
    if (conf_.early_stop()) {
      CHECK(conf_.val_data().size()) << "early stop needs validation dataset";
    }
    Init(conf);
  }
  virtual ~FMScheduler() { }

  virtual bool Stop(const Progress& cur, const Progress& prev, bool train) {
    double cur_objv = cur.objv() / cur.num_ex();
    if (train) {
      if (conf_.has_max_objv() && cur_objv > conf_.max_objv()) {
        return true;
      }
    } else {
      double diff = pre_val_objv_ - cur_objv;
      pre_val_objv_ = cur_objv;
      if (conf_.early_stop() && diff < conf_.min_objv_decr()) {
        std::cout << "the decrease of validation objective "
                  << "is smaller than the minimal requirement: "
                  << diff << " vs " << conf_.min_objv_decr()
                  << std::endl;
        return true;
      }
    }
    return false;
  }
 private:
  Config conf_;
  double pre_val_objv_ = 100;
};

}  // namespace fm
}  // namespace dmlc

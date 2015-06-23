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

  Progress() : VectorProgress(3, 6) {}
  virtual ~Progress() { }


  /// head string for printing
  virtual std::string HeadStr() {
    return "#example delta #ex    |w|_0       objv       AUC    accuracy";
  }

  /// string for printing
  virtual std::string PrintStr(const IProgress* const prev) {
    Progress* const p = (Progress* const) prev;
    if (num_ex() == 0) return "";
    char buf[60];
    snprintf(buf, 60, "%7.2g  %7.2g  %11.6g  %8.6lf  %8.6lf  %8.6lf",
             (double)(p->num_ex() + num_ex()),
             (double)num_ex(),
             (double)(p->nnz_w() + nnz_w()),
             objv() / num_ex(), auc() / count(), acc() / count());
    return std::string(buf);
  }

  // mutator
  double& objv() { return fvec_[0]; }
  double& acc() { return fvec_[1]; }
  double& auc() { return fvec_[2]; }
  double& logloss() { return fvec_[3]; }
  double& weight2() { return fvec_[4]; }
  double& wdelta2() { return fvec_[5]; }

  size_t& count() { return ivec_[0]; }

  size_t& num_ex() { return ivec_[1]; }
  size_t& nnz_w() { return ivec_[2]; }

  size_t num_ex() const { return ivec_[1]; }
  size_t nnz_w() const { return ivec_[2]; }
};

class FMScheduler : public solver::AsyncSGDScheduler<Progress> {
 public:
  FMScheduler(const Config& conf) {
    CHECK(conf.has_train_data());
    train_data_        = conf.train_data();
    val_data_          = conf.val_data();
    data_format_       = conf.data_format();
    num_part_per_file_ = conf.num_parts_per_file();
    max_data_pass_     = conf.max_data_pass();
    disp_itv_          = conf.disp_itv();
  }
  virtual ~FMScheduler() { }
};

}  // namespace fm
}  // namespace dmlc

#pragma once
#include "fm.h"
#include "base/localizer.h"

namespace dmlc {
namespace fm {

////////////////////////////////////////////////////////////
//  objective
////////////////////////////////////////////////////////////

class Objective {
 public:
  Objective(const RowBlock<unsigned>& data,
            const std::vector<Real>& w,
            const std::vector<int>& w_siz) {

  }

  ~Objective() { }
  /*! \brief evaluate the objective value */
  Real Objv() {
    return 0;
  }

  /*! \brief compute the gradients */
  void CalcGrad(std::vector<Real>* grad, std::vector<int>* grad_siz) {
  }

 private:

};

////////////////////////////////////////////////////////////
//  sgd solver
////////////////////////////////////////////////////////////

class FMWorker : public solver::AsyncSGDWorker {
 public:
  FMWorker(const Config& conf) : conf_(conf) { }
  virtual ~FMWorker() { }

 protected:

  virtual void ProcessMinibatch(const Minibatch& mb, bool train) {

    auto data = new dmlc::data::RowBlockContainer<unsigned>();
    auto feaid = std::make_shared<std::vector<FeaID>>();
    auto feacnt = std::make_shared<std::vector<Real>>();

    Localizer<FeaID> lc;
    lc.Localize(mb, data, feaid.get(), feacnt.get());

    ps::SyncOpts pull_w_opts;

    if (train) {
      // push the feature count to the servers
      ps::SyncOpts push_cnt_opts;
      SetFilters(true, &push_cnt_opts);
      int t = server_.ZPush(feaid, feacnt, push_cnt_opts);
      pull_w_opts.deps.push_back(t);
    }

    // pull the weight from the servers
    auto val = new std::vector<Real>();
    auto val_siz = new std::vector<int>();

    // this callback will be called when the weight has been actually pulled back
    pull_w_opts.callback = [this, data, feaid, val, val_siz, train]() {
      // eval the progress
      Objective obj(data->GetBlock(), *val, *val_siz);
      Progress prog;
      // report progress to the scheduler
      Report(&prog);

      // monitor_.Update(local->label.size(), loss);

      if (train) {
        // calculate and push the gradients
        obj.CalcGrad(val, val_siz);

        ps::SyncOpts push_grad_opts;
        // filters to reduce network traffic
        SetFilters(true, &push_grad_opts);
        // this callback will be called when the gradients have been actually pushed
        push_grad_opts.callback = [this]() { FinishMinibatch(); };
        server_.ZVPush(feaid,
                       std::shared_ptr<std::vector<Real>>(val),
                       std::shared_ptr<std::vector<int>>(val_siz),
                       push_grad_opts);
      } else {
        FinishMinibatch();
        delete val;
        delete val_siz;
      }
      delete data;
    };

    // filters to reduce network traffic
    SetFilters(false, &pull_w_opts);
    server_.ZVPull(feaid, val, val_siz, pull_w_opts);
  }

 private:
  void SetFilters(bool push, ps::SyncOpts* opts) {
    if (conf_.fixed_bytes() > 0) {
      opts->AddFilter(ps::Filter::FIXING_FLOAT)->set_num_bytes(conf_.fixed_bytes());
    }
    if (conf_.key_cache()) {
      opts->AddFilter(ps::Filter::KEY_CACHING)->set_clear_cache(push);
    }
    if (conf_.msg_compression()) {
      opts->AddFilter(ps::Filter::COMPRESSING);
    }
  }
  Config conf_;
  ps::KVWorker<Real> server_;
};

}  // namespace fm
}  // namespace dmlc

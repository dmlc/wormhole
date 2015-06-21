#pragma once
#include "fm.h"

namespace dmlc {
namespace fm {

////////////////////////////////////////////////////////////
//  objective
////////////////////////////////////////////////////////////

class Objective {
 public:
  void Init() {

  }
  /*! \brief evaluate the objective value */
  Real Objv() {
    return 0;
  }

  /*! \brief compute the gradients */
  void CalcGrad() {

  }

 private:

};

////////////////////////////////////////////////////////////
//  sgd solver
////////////////////////////////////////////////////////////

class FMWorker : public solver::AsyncSGDWorker {
 public:
  FMWorker(const Config& conf) { }
  virtual ~FMWorker() { }

 protected:
  virtual void ProcessMinibatch(const Minibatch& mb) {
    // Minibatch* local = new Minibatch();
    // shared_ptr<vector<FeaID> > feaid(new vector<FeaID>());
    // shared_ptr<vector<Real> > feacnt(new vector<Real>());

    // Localizer<FeaID> lc; lc.Localize(global, local, feaid.get(), feacnt.get());

    // // wait for data consistency
    // WaitMinibatch(max_delay);

    // // push the feature count to the servers
    // ps::SyncOpts push_opts;
    // SetFilters(true, &push_opts);
    // int push_t = server_.ZPush(feaid, feacnt, push_opts);

    // // pull the weight from the servers
    // vector<Real>* val = new vector<Real>();
    // vector<int>* len_val = new vector<Real>(feaid.get()->size());
    // ps::SyncOpts opts;

    // // this callback will be called when the weight has been actually pulled back
    // opts.callback = [this, local, feaid, val, len_val, type]() {
    //   // eval the progress
    //   Objective obj;
    //   obj.Init(local->GetBlock(), *val, *len_val);

    //   // monitor_.Update(local->label.size(), loss);

    //   if (type == Workload::TRAIN) {
    //     // reporting from time to time
    //     reporter_.Report(0, &monitor_.prog);

    //     // calculate and push the gradients
    //     loss->CalcGrad(val, len_val);
    //     ps::SyncOpts opts;
    //     // this callback will be called when the gradients have been actually pushed
    //     opts.callback = [this]() { FinishMinibatch(); };
    //     // filters to reduce network traffic
    //     SetFilters(true, &opts);
    //     server_.ZPush(feaid, shared_ptr<vector<Real>>(val),
    //                   shared_ptr<vector<int>>(len_val), opts);
    //   } else {
    //     // don't need to cal grad for evaluation task
    //     FinishMinibatch();
    //     delete buf;
    //   }
    //   delete local;
    //   delete loss;
    // };

    // // filters to reduce network traffic
    // SetFilters(false, &opts);
    // opts.deps.push_push(push_t);
    // server_.ZPull(feaid, val, val_len, opts);

  }
};

}  // namespace fm
}  // namespace dmlc

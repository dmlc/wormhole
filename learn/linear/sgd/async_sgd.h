/**
 * @file   async_sgd.h
 * @brief  Asynchronous stochastic gradient descent to solve linear methods.
 */
#include "proto/config.pb.h"
#include "proto/sys.pb.h"
#include "dmlc/timer.h"
#include "base/minibatch_iter.h"
#include "base/arg_parser.h"
#include "base/localizer.h"
#include "base/loss.h"
#include "sgd/sgd_server_handle.h"
#include "base/dist_monitor.h"

#include "base/utils.h"
#include "ps.h"
#include "ps/app.h"
namespace dmlc {
namespace linear {

using FeaID = ps::Key;
using Real = float;

// commands
static const int kRequestWorkload = 1;

/***************************************
 * \brief The scheduler node
 **************************************/

class AsyncSGDScheduler : public ps::App {
 public:
  AsyncSGDScheduler(const Config& conf) : conf_(conf) {

  }
  virtual ~AsyncSGDScheduler() { }

  virtual void Run() {

  }
 private:
  Config conf_;
};

/***************************************
 * \brief A server node
 **************************************/

class AsyncSGDServer : public ps::App {
 public:
  AsyncSGDServer(const Config& conf) : conf_(conf), monitor_(conf_.show_prog())  {
    Init();
  }
  virtual ~AsyncSGDServer() { }

  virtual void Run() { }  // empty
 private:
  void Init() {
    auto algo = conf_.algo();
    if (algo == Config::FTRL) {
      ps::KVServer<Real, FTRLHandle<FeaID, Real>, 3> ftrl;
      auto& updt = ftrl.handle();
      if (conf_.has_lr_eta()) updt.alpha = conf_.lr_eta();
      if (conf_.has_lr_beta()) updt.beta = conf_.lr_beta();
      if (conf_.lambda_size() > 0) updt.lambda1 = conf_.lambda(0);
      if (conf_.lambda_size() > 1) updt.lambda2 = conf_.lambda(1);
      updt.tracker = &monitor_;
      ftrl.Run();
    } else {
      LOG(FATAL) << "unknown algo: " << algo;
    }
  }
  Config conf_;
  DistModelMonitor monitor_;
};

/***************************************
 * \brief A worker node
 **************************************/
class AsyncSGDWorker : public ps::App {
 public:
  AsyncSGDWorker(const Config& conf) : conf_(conf), reporter_(conf_.show_prog()) { }
  virtual ~AsyncSGDWorker() { }

  virtual void Run() {
    while (true) {
      using namespace ps;
      // request one data file from the scheduler
      Task task; task.set_cmd(kRequestWorkload);
      Wait(Submit(task, SchedulerID()));
      std::string file = LastResponse()->task.msg();
      if (file.empty()) {
        LOG(INFO) << MyNodeID() << ": all workloads are done";
        break;
      }
      Process(file);
    }
  }
 private:
  void Process(const std::string file_str) {
    File file; CHECK(file.ParseFromString(file_str));
    LOG(INFO) << ps::MyNodeID() << ": start to process " << file.ShortDebugString();

    dmlc::data::MinibatchIter<FeaID> reader(
        file.file().c_str(), file.k(), file.n(),
        conf_.data_format().c_str(), conf_.minibatch());

    reader.BeforeFirst();
    while (reader.Next()) {
      using std::vector;
      using std::shared_ptr;
      using Minibatch = dmlc::data::RowBlockContainer<unsigned>;

      // localize the minibatch
      auto global = reader.Value();
      Minibatch* local = new Minibatch();
      shared_ptr<vector<FeaID> > feaid(new vector<FeaID>());
      Localizer<FeaID> lc; lc.Localize(global, local, feaid.get());

      // pull the weight
      vector<Real>* buf = new vector<Real>(feaid.get()->size());
      ps::SyncOpts opts;
      bool is_train = file.train();
      opts.callback = [this, local, feaid, buf, is_train]() {
        // eval
        auto loss = CreateLoss<real_t>(conf_.loss());
        loss->Init(local->GetBlock(), *buf);
        monitor_.Update(local->label.size(), loss);
        reporter_.Report(&monitor_.prog);

        // calc and push grad
        if (is_train) {
          loss->CalcGrad(buf);
          ps::SyncOpts opts;
          server_.ZPush(feaid, shared_ptr<vector<Real>>(buf), opts);
        } else {
          delete buf;
        }
        delete local;
        delete loss;
      };
      server_.ZPull(feaid, buf, opts);
    }
    LOG(INFO) << ps::MyNodeID() << ": finished";
  }

  Config conf_;
  ps::KVWorker<Real> server_;
  WorkerMonitor monitor_;
  TimeReporter reporter_;
};


}  // namespace linear
}  // namespace dmlc

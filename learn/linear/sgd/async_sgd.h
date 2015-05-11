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
#include "base/workload_pool.h"

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
    CHECK(conf_.has_train_data());
    for (int i = 0; i < conf_.max_data_pass(); ++i) {
      pool_.Add(conf_.train_data(), conf_.num_parts_per_file());
    }
  }
  virtual ~AsyncSGDScheduler() { }

  virtual void ProcessRequest(ps::Message* request) {
    if (request->task.cmd() == kRequestWorkload) {
      RequestWorkload(request);
    }
  }

  virtual void Run() {
    double t = GetTime();
    int itv = (int) (conf_.show_prog() * 1000000);
    usleep(itv / 2);
    size_t num_ex = 0;

    while (!done_) {
      usleep(itv);
      Progress prog; monitor_.Get(0, &prog);
      monitor_.Clear(0);
      num_ex += prog.num_ex();
      std::cout << GetTime() - t << " sec, "
                << "#ex " << num_ex
                << prog.PrintStr() << std::endl;
    }
    // TODO save model

  }
 private:
  void RequestWorkload(ps::Message* req) {
    // a simple version
    pool_.Finish(req->sender);
    Files files; pool_.Get(req->sender, &files);
    files.set_train(true);
    std::string msg; files.SerializeToString(&msg);
    ps::Task res; res.set_msg(msg);
    Reply(req, res);

    if (pool_.IsFinished()) done_ = true;
  }

  Config conf_;
  WorkloadPool pool_;
  bool done_ = false;
  // int cur_iter_ = 0;
  // bool is_train_ = true;
  ps::MonitorMaster<Progress> monitor_;
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
      ftrl.set_sync_val_len(1);
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
    Files files; CHECK(files.ParseFromString(file_str));
    LOG(INFO) << ps::MyNodeID() << ": start to process " << files.ShortDebugString();
    CHECK_EQ(files.file_size(), 1);
    File file = files.file(0);

    dmlc::data::MinibatchIter<FeaID> reader(
        file.file().c_str(), file.k(), file.n(),
        conf_.data_format().c_str(), conf_.minibatch());

    num_mb_fly_ = num_mb_done_ = 0;
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
      bool is_train = files.train();
      opts.callback = [this, local, feaid, buf, is_train]() {
        // eval
        auto loss = CreateLoss<real_t>(conf_.loss());
        loss->Init(local->GetBlock(), *buf);
        monitor_.Update(local->label.size(), loss);
        // LOG(ERROR) << monitor_.prog.PrintStr();
        reporter_.Report(0, &monitor_.prog);

        // calc and push grad
        if (is_train) {
          loss->CalcGrad(buf);
          ps::SyncOpts opts;
          opts.callback = [this]() {
            // wake the main thread
            mb_mu_.lock(); -- num_mb_fly_; ++ num_mb_done_; mb_mu_.unlock();
            mb_cond_.notify_one();
          };
          server_.ZPush(feaid, shared_ptr<vector<Real>>(buf), opts);
        } else {
          delete buf;
        }
        delete local;
        delete loss;
      };
      server_.ZPull(feaid, buf, opts);

      // wait for data consistency
      std::unique_lock<std::mutex> lk(mb_mu_);
      ++ num_mb_fly_;
      mb_cond_.wait(lk, [this] {return conf_.max_delay() <= num_mb_fly_;});
    }

    // wait untill all are done
    std::unique_lock<std::mutex> lk(mb_mu_);
    mb_cond_.wait(lk, [this] {return num_mb_fly_ <= 0;});
    LOG(INFO) << ps::MyNodeID() << ": finished";
  }

  Config conf_;
  ps::KVWorker<Real> server_;
  WorkerMonitor monitor_;
  TimeReporter reporter_;

  int num_mb_fly_;
  int num_mb_done_;
  std::mutex mb_mu_;
  std::condition_variable mb_cond_;

};


}  // namespace linear
}  // namespace dmlc

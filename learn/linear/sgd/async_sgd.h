/**
 * @file   async_sgd.h
 * @brief  Asynchronous stochastic gradient descent to solve linear methods.
 */
#include "proto/config.pb.h"
#include "proto/workload.pb.h"
#include "dmlc/timer.h"
#include "base/minibatch_iter.h"
#include "base/arg_parser.h"
#include "base/localizer.h"
#include "base/loss.h"
#include "sgd/sgd_server_handle.h"
#include "sgd/delay_tol_handle.h"
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
static const int kProcess = 1;
static const int kSaveModel = 2;

/*****************************************************************************
 * \brief A worker node, which takes a part of training data and calculate the
 * gradients
 *****************************************************************************/

class AsyncSGDWorker : public ps::App {
 public:
  AsyncSGDWorker(const Config& conf) : conf_(conf), reporter_(conf_.disp_itv()) { }
  virtual ~AsyncSGDWorker() { }

  // process request from the scheduler node
  virtual void ProcessRequest(ps::Message* request) {
    int cmd = request->task.cmd();
    if (cmd == kProcess) {
      Workload wl; CHECK(wl.ParseFromString(request->task.msg()));
      if (wl.file_size() < 1) return;
      Process(wl.file(0), wl.type());
      if (wl.type() != Workload::TRAIN) {
        // return the progress
        std::string prog_str; monitor_.prog.Serialize(&prog_str);
        ps::Task res; res.set_msg(prog_str);
        Reply(request, res);
      }
    }
  }

 private:
  void Process(const File& file, Workload::Type type) {
    // use a large minibatch size and max_delay for val or test tasks
    int mb_size = type == Workload::TRAIN ? conf_.minibatch() :
                  std::max(conf_.minibatch()*10, 100000);
    int max_delay = type == Workload::TRAIN ? conf_.max_delay() : 100000;
    num_mb_fly_ = num_mb_done_ = 0;

    LOG(INFO) << ps::MyNodeID() << ": start to process " << file.ShortDebugString();
    dmlc::data::MinibatchIter<FeaID> reader(
        file.file().c_str(), file.k(), file.n(),
        conf_.data_format().c_str(), mb_size);
    reader.BeforeFirst();
    while (reader.Next()) {
      using std::vector;
      using std::shared_ptr;
      using Minibatch = dmlc::data::RowBlockContainer<unsigned>;

      // used for debug IO performance
      if (conf_.test_io()) {
        monitor_.prog.num_ex() += reader.Value().size;
        reporter_.Report(0, &monitor_.prog);
        continue;
      }

      // find all feature id in this minibatch, and convert it to a more compact format
      auto global = reader.Value();
      Minibatch* local = new Minibatch();
      shared_ptr<vector<FeaID> > feaid(new vector<FeaID>());
      Localizer<FeaID> lc; lc.Localize(global, local, feaid.get());

      // pull the weight from the servers
      vector<Real>* buf = new vector<Real>(feaid.get()->size());
      ps::SyncOpts opts;

      // this callback will be called when the weight has been actually pulled back
      opts.callback = [this, local, feaid, buf, type]() {
        // eval the progress
        auto loss = CreateLoss<real_t>(conf_.loss());
        loss->Init(local->GetBlock(), *buf);
        monitor_.Update(local->label.size(), loss);

        if (type == Workload::TRAIN) {
          // reporting from time to time
          reporter_.Report(0, &monitor_.prog);

          // calculate and push the gradients
          loss->CalcGrad(buf);
          ps::SyncOpts opts;
          // this callback will be called when the gradients have been actually pushed
          opts.callback = [this]() { FinishMinibatch(); };
          // filters to reduce network traffic
          opts.AddFilter(ps::Filter::FIXING_FLOAT)->set_num_bytes(1);
          opts.AddFilter(ps::Filter::KEY_CACHING)->set_clear_cache(true);
          opts.AddFilter(ps::Filter::COMPRESSING);
          server_.ZPush(feaid, shared_ptr<vector<Real>>(buf), opts);
        } else {
          // don't need to cal grad for evaluation task
          FinishMinibatch();
          delete buf;
        }
        delete local;
        delete loss;
      };

      // filters to reduce network traffic
      opts.AddFilter(ps::Filter::FIXING_FLOAT)->set_num_bytes(1);
      opts.AddFilter(ps::Filter::KEY_CACHING);
      opts.AddFilter(ps::Filter::COMPRESSING);
      server_.ZPull(feaid, buf, opts);

      // wait for data consistency
      std::unique_lock<std::mutex> lk(mb_mu_);
      ++ num_mb_fly_;
      mb_cond_.wait(lk, [this, max_delay] {return max_delay >= num_mb_fly_;});
      LOG(INFO) << num_mb_fly_;
    }

    // wait untill all are done
    std::unique_lock<std::mutex> lk(mb_mu_);
    mb_cond_.wait(lk, [this] {return num_mb_fly_ <= 0;});
    LOG(INFO) << ps::MyNodeID() << ": finished";
  }

  void FinishMinibatch() {
    // wake the main thread
    mb_mu_.lock();
    -- num_mb_fly_;
    ++ num_mb_done_;
    mb_mu_.unlock();
    mb_cond_.notify_one();
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

/**************************************************************************
 * \brief A server node, which maintains a part of the model, and update the
 * model once received gradients from workers
 **************************************************************************/
class AsyncSGDServer : public ps::App {
 public:
  AsyncSGDServer(const Config& conf)
      : model_(NULL), conf_(conf), monitor_(conf_.disp_itv())  {
    Init();
  }
  virtual ~AsyncSGDServer() { }

  virtual void ProcessRequest(ps::Message* request) {
    if (request->task.cmd() == kSaveModel) {
      if (conf_.has_model_out()) {
        model_->SaveModel(conf_.model_out());
      }
    }
  }

 private:
  template <typename Handle>
  void InitHandle(Handle* h) {
    L1L2<Real> l1l2;
    if (conf_.lambda_size() > 0) l1l2.set_lambda1(conf_.lambda(0));
    if (conf_.lambda_size() > 1) l1l2.set_lambda2(conf_.lambda(1));
    h->penalty = l1l2;

    if (conf_.has_lr_eta()) h->alpha = conf_.lr_eta();
    if (conf_.has_lr_beta()) h->beta = conf_.lr_beta();

    h->tracker = &monitor_;
  }

  void Init() {
    auto algo = conf_.algo();
    if (algo == Config::SGD) {
      ps::KVServer<Real, SGDHandle<FeaID, Real>, 1> sgd;
      InitHandle(&sgd.handle());
      model_ = sgd.Run();
    } else if (algo == Config::ADAGRAD) {
      ps::KVServer<Real, AdaGradHandle<FeaID, Real>, 2> adagrad;
      adagrad.set_sync_val_len(1);
      InitHandle(&adagrad.handle());
      model_ = adagrad.Run();
    } else if (algo == Config::FTRL) {
      ps::KVServer<Real, FTRLHandle<FeaID, Real>, 3> ftrl;
      ftrl.set_sync_val_len(1);
      InitHandle(&ftrl.handle());
      model_ = ftrl.Run();
    } else if (algo == Config::DT_SGD) {
      ps::KVServer<Real, DTSGDHandle<FeaID, Real>, 1> sgd;
      InitHandle(&sgd.handle());
      model_ = sgd.Run();
    } else {
      LOG(FATAL) << "unknown algo: " << algo;
    }
  }
  ps::KVStore* model_;
  Config conf_;
  DistModelMonitor monitor_;
};

/**************************************************************************
 * \brief The scheduler node, which issues workloads to workers/servers, in
 * charge of fault tolerance, and also print the progress
 **************************************************************************/
class AsyncSGDScheduler : public ps::App {
 public:
  AsyncSGDScheduler(const Config& conf) : conf_(conf) { }
  virtual ~AsyncSGDScheduler() { }

  virtual void ProcessResponse(ps::Message* response) {
    if (response->task.cmd() == kProcess) {
      auto id = response->sender;
      if (!response->task.msg().empty()) {
        Progress p;
        p.Parse(response->task.msg());
        prog_.Merge(p);
      }
      pool_.Finish(id);
      Workload wl; pool_.Get(id, &wl);
      if (wl.file_size() > 0) SendWorkload(id, wl);
    }
  }

  virtual void Run() {
    printf("waiting %d workers and %d servers are connected\n",
           ps::NumWorkers(), ps::NumServers());
    // wait nodes are ready
    ps::App::Run();

    CHECK(conf_.has_train_data());
    double t = GetTime();
    size_t num_ex = 0;
    int64_t nnz_w = 0;
    for (int i = 0; i < conf_.max_data_pass(); ++i) {
      printf("training #iter = %d\n", i);
      // train
      pool_.Clear();
      pool_.Add(conf_.train_data(), conf_.num_parts_per_file(), 0, Workload::TRAIN);
      Workload wl; SendWorkload(ps::kWorkerGroup, wl);

      printf("time(sec)  #example  delta #ex    |w|_1   %s\n", prog_.HeadStr().c_str());
      sleep(1);
      while (!pool_.IsFinished()) {
        sleep((int) conf_.disp_itv());
        Progress prog; monitor_.Get(0, &prog);
        monitor_.Clear(0);
        if (prog.Empty()) continue;
        num_ex += prog.num_ex();
        nnz_w += prog.nnz_w();
        printf("%7.0lf  %10.5g  %8ld  %9ld  %s\n",
               GetTime() - t, (double)num_ex, prog.num_ex(), nnz_w,
               prog.PrintStr().c_str());
      }

      // val
      if (!conf_.has_val_data()) continue;
      printf("validation #iter = %d\n", i);
      pool_.Clear();
      pool_.Add(conf_.val_data(), conf_.num_parts_per_file(), 0, Workload::VAL);
      SendWorkload(ps::kWorkerGroup, wl);

      while (!pool_.IsFinished()) { sleep(1); }

      printf("%7.1lf sec, #val %.3g, %s\n",
             GetTime() - t, (double)prog_.num_ex(), prog_.PrintStr().c_str());
      prog_.Clear();
    }

    printf("saving model\n");
    ps::Task task; task.set_cmd(kSaveModel);
    Wait(Submit(task, ps::kServerGroup));
  }
 private:
  void SendWorkload(const std::string id, const Workload& wl) {
    std::string wl_str; wl.SerializeToString(&wl_str);
    ps::Task task; task.set_msg(wl_str);
    task.set_cmd(kProcess);
    Submit(task, id);
  }

  Config conf_;
  WorkloadPool pool_;
  bool done_ = false;
  Progress prog_;
  ps::MonitorMaster<Progress> monitor_;
};


}  // namespace linear
}  // namespace dmlc

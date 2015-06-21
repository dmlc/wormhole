/**
 * @file   async_sgd.h
 * @brief  A template to implement async sgd in parameter server
 */
#include "ps.h"
#include "ps/app.h"
#include "base/progress.h"
#include "base/dist_monitor.h"
#include "base/workload_pool.h"

namespace dmlc {
namespace solver {

// commands
static const int kProcess = 1;
static const int kSaveModel = 2;

/**************************************************************************
 * \brief The scheduler node, which issues workloads to workers/servers, in
 * charge of fault tolerance, and also prints the progress
 **************************************************************************/
template <typename Progress>
class AsyncSGDScheduler : public ps::App {
 protected:
  std::string train_data_;
  std::string val_data_;
  std::string data_format_;
  int save_model_ = 0;

  int num_part_per_file_ = 10;
  int max_data_pass_ = 1;
  int cur_data_pass_ = 0;
  int disp_itv_ = 1;

  virtual bool Stop(const Progress& cur, const Progress& prev) {
    return false;
  }

 public:
  AsyncSGDScheduler() {
    sys_.manager().AddNodeFailureHandler([this](const std::string& id) {
        pool_.Reset(id);
      });
  }
  virtual ~AsyncSGDScheduler() { }

  virtual void ProcessResponse(ps::Message* response) {
    if (response->task.cmd() == kProcess) {
      auto id = response->sender;
      pool_.Finish(id);
      Workload wl; pool_.Get(id, &wl);
      if (wl.file.size() > 0) SendWorkload(id, wl);
    }
  }

  virtual bool Run() {
    double t = GetTime();
    for (int i = 0; i < max_data_pass_; ++i) {
      cur_data_pass_ = i;
      printf("training #iter = %d\n", i);
      bool exit = !Iterate(true, t);

      printf("validating #iter = %d\n", i);
      Iterate(false, t);

      if (exit) break;
    }

    if (save_model_) {
      printf("saving model");
      ps::Task task; task.set_cmd(kSaveModel);
      Wait(Submit(task, ps::kServerGroup));
    }
    printf("async_sgd done!\n");
    return true;
  }

 private:
  bool Iterate(bool is_train, double start_time) {
    bool stop = false;
    pool_.Clear();
    if (is_train) {
      if (train_data_.empty()) return;
      pool_.Add(train_data_, data_format_, num_part_per_file_,  Workload::TRAIN);
    } else {
      if (val_data_.empty()) return;
        pool_.Add(val_data_, data_format_, num_part_per_file_, Workload::VAL);
    }
    Workload wl; SendWorkload(ps::kWorkerGroup, wl);
    printf(" sec %s\n", prog_.HeadStr().c_str());

    Progress cur;
    while (!pool_.IsFinished()) {
      sleep(disp_itv_);
      if (is_train) {
        // continous print
        monitor_.Get(&cur); monitor_.Clear();
        if (cur.Empty()) continue;
        printf("%5.0lf  %s\n", GetTime() - start_time, cur.PrintStr(&prog_));
        if (Stop(cur, prog_)) {
          stop = true;
          pool_.ClearRemain();
        }
        prog_.Merge(&cur);
      }
    }

    if (!is_train) {
      // get cur
      monitor_.Get(&cur); monitor_.Clear();
      printf("%5.0lf  %s\n", GetTime() - start_time, cur.PrintStr(&prog_));
      prog_.Merge(&cur);
    }
  }

  void SendWorkload(const std::string id, const Workload& wl) {
    std::string wl_str; // TODO wl.SerializeToString(&wl_str);
    ps::Task task; task.set_msg(wl_str);
    task.set_cmd(kProcess);
    Submit(task, id);
  }

  WorkloadPool pool_;
  bool done_ = false;
  Progress prog_;
  ProgressMonitor<Progress> monitor_;
};

/**************************************************************************
 * \brief A server node, which maintains a part of the model, and update the
 * model once received gradients from workers
 **************************************************************************/
class AsyncSGDServer : public ps::App {
 protected:
  virtual void SaveModel() = 0;

  /**
   * \brief Report the progress to the scheduler
   */
  void Report(const IProgress* const prog) {
    reporter_.Report(prog);
  }

  int report_itv_ = 1;
 public:
  AsyncSGDServer() {
    reporter_.set_report_itv(report_itv_);
  }
  virtual ~AsyncSGDServer() { }

  virtual void ProcessRequest(ps::Message* request) {
    if (request->task.cmd() == kSaveModel) {
      SaveModel();
    }
  }

 private:
  ProgressReporter reporter_;
};

/*****************************************************************************
 * \brief A worker node, which takes a part of training data and calculate the
 * gradients
 *****************************************************************************/
class AsyncSGDWorker : public ps::App {
  /// for applications
 protected:
  // feature id type
  using FeaID = ps::Key;
  // a minibatch data
  using Minibatch = dmlc::RowBlock<FeaID>;

  /**
   * \brief Process one minibatch
   */
  virtual void ProcessMinibatch(const Minibatch& mb) = 0;

  /**
   * \brief Mark one minibatch is finished
   * one must call this function when the minibatch is actually done
   */
  void FinishMinibatch();

  /**
   * \brief Report the progress to the scheduler
   */
  void Report(const IProgress *const prog) {
    reporter_.Report(prog);
  }

  int minibatch_size_ = 10000;
  int max_delay_ = 4;

  // for validation test
  int val_minibatch_size_ = 1000000;
  int val_max_delay_ = 10;

  int report_itv_ = 1;

  /// implement system APIs
 public:
  AsyncSGDWorker() { }
  virtual ~AsyncSGDWorker() { }

  // process request from the scheduler node
  virtual void ProcessRequest(ps::Message* request) {
    int cmd = request->task.cmd();
    if (cmd == kProcess) {
      Workload wl; // CHECK(wl.ParseFromString(request->task.msg()));
      if (wl.file.size() < 1) return;
      Process(wl.file[0], wl.type);
    }
  }

 private:
  void Process(const Workload::File& file, Workload::Type type) {
    LOG(INFO) << ps::MyNodeID() << ": start to process " << file.ShortDebugString();

    int mb_size = type == Workload::TRAIN ? minibatch_size_ : val_minibatch_size_;
    int max_delay = type == Workload::TRAIN ? max_delay_ : val_max_delay_;

    dmlc::data::MinibatchIter<FeaID> reader(
        file.filename.c_str(), file.k, file.n, file.format.c_str(), mb_size);
    reader.BeforeFirst();
    while (reader.Next()) {
      // wait for data consistency
      WaitMinibatch(max_delay);

      ProcessMinibatch(reader.Value());

      mb_mu_.lock(); ++ num_mb_fly_; mb_mu_.unlock();

      LOG(INFO) << "#minibatches on processing: " << num_mb_fly_;
    }
  }

  // wait if the currenta number of on processing minibatch > num
  inline void WaitMinibatch(int num) {
    std::unique_lock<std::mutex> lk(mb_mu_);
    mb_cond_.wait(lk, [this, num] {return num_mb_fly_ <= num;});
  }

  int num_mb_fly_;
  int num_mb_done_;
  std::mutex mb_mu_;
  std::condition_variable mb_cond_;

  ProgressReporter reporter_;
};

void AsyncSGDWorker::FinishMinibatch() {
  // wake the main thread
  mb_mu_.lock();
  -- num_mb_fly_;
  ++ num_mb_done_;
  mb_mu_.unlock();
  mb_cond_.notify_one();
}



}  // namespace solver
}  // namespace dmlc

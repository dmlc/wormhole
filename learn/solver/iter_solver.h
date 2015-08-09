/**
 * @file   iter_solver.h
 * @brief  Template for an iterate solver
 */
#include "ps.h"
#include "base/string_stream.h"
#include "base/workload.h"
#include "base/workload_pool.h"
namespace dmlc {
namespace solver {

using Progress = std::vector<double>;

/**
 * \brief encode/decode a command
 */
struct IterCmd {
  IterCmd() {}
  IterCmd(int c) : cmd(c) {}

  // mutators
  void set_iter(int iter) { cmd += iter << 5; }
  void set_process() { cmd |= 1; }
  void set_load_model() { cmd |= 1<<1; }
  void set_save_model() { cmd |= 1<<2; }

  // accessors
  bool process() const { return cmd & 1; }
  bool load_model() const { return cmd & 1<<1; }
  bool save_model() const { return cmd & 1<<2; }
  bool iter() const { return cmd >> 5; }

  int cmd = 0;
};

/**
 * \brief the scheduler node for an iterate solver
 */
class IterScheduler : public ps::App {
 protected:

  /// \brief training data, supports regular expression
  std::string train_data_;

  /// \brief validation data, supports regular expression
  std::string val_data_;

  /// \brief libsvm, crb, ...
  std::string data_format_;

  /// \brief  virtually partition a file into n parts for better loadbalance.
  int num_parts_per_file_ = 10;

  /// \brief batch or online assignment
  ///
  /// assume there are n parts ( = num_files_matched * num_parts_per_file_ )
  ///
  /// * batch mode. the scheduler gives a worker \a n / \a num_workers parts
  /// each time. This assignment does not change between iterations
  ///
  /// * online mode. The scheduler gives one part to a worker each time. The
  /// assignment is random, which is depended on the speeds of each worker
  bool batch_ = false;

  /// \brief whether shuffle the data parts before assigning
  bool shuffle_ = false;

  /// \brief mark a worker as the straggler if it uses \a straggler_ times
  /// longer than average. on the online mode, the scheduler will reassign that
  /// workerload to another worker.
  int straggler_ = 3;

  /// \brief whether allow each worker to tell the scheduler which file it can
  /// access
  ///
  /// If true, each worker matches the files it can access and then report to
  /// the scheduler. It is useful when the data have been dispatched into
  /// workers' local disks. it can reduce the cost to access data remotely
  bool use_worker_local_data_ = false;

  /// \brief model input.
  std::string model_in_;

  /// \brief load model from a particualr iteration. if 0, then load the last
  /// iteration
  int load_iter_ = 0;

  /// \brief model output
  std::string model_out_;

  /// \brief save model for every k iterations. if 0, then only save the last
  /// iteration
  int save_iter_ = 0;

  /// \brief print the progress for every k seconds. only valid for the online model
  int print_sec_ = 1;

  /// \brief the maximal number of data passes
  int max_data_pass_ = 1;

  /// \brief if set, then run a prediction task
  std::string predict_out_;

  // runtime info

  /// \brief the current data pass, starting from 0.
  int cur_data_pass_;

  /// \brief the current task, training or validation?
  Workload::Type cur_task_;

  /**
   * \brief a user-defined stop criteria. stop the system if returns true
   *
   * @param prog the current progress
   * @param train in training or validation
   */
  virtual bool Stop(const Progress& prog, bool train) {
    return false;
  }

  /**
   * \brief returns the header string for the progress
   */
  virtual std::string ProgHeader() const { return ""; }

  /**
   * \brief returns the string for the progress. empty string means no progress
   */
  virtual std::string ProgString(const Progress& prog) const { return ""; }

  // implementation
 public:

  static const int kProcess = 1;
  static const int kSaveModel = 2;
  static const int kLoadModel = 3;
  static const int kMaxNumCmd = 10;

  IterScheduler() {
    sys_.manager().AddNodeFailureHandler([this](const std::string& id) {
        pool_.Reset(id);
      });
  }
  virtual ~IterScheduler() { }

  /// \brief run iterations
  virtual bool Run() {
    printf("Connected %d servers and %d workers\n",
           ps::NodeInfo::NumServers(), ps::NodeInfo::NumWorkers());

    start_time_ = GetTime();

    bool is_predict = predict_out_.size();
    if (is_predict) {
      CHECK(model_in_.size()) << "should provide model_in for predicting";
    }

    if (model_in_.size()) {
      if (load_iter_ > 0) {
        printf("Loading model from #iter = %d\n", load_iter_);
        cur_data_pass_ = load_iter_;
      } else {
        printf("Loading the last model\n");
        cur_data_pass_ = 0; // wrong number...
      }
      IterCmd cmd; cmd.set_load_model(); cmd.set_iter(load_iter_);
      ps::Task task; task.set_cmd(cmd.cmd); task.set_msg(model_in_);
      Wait(Submit(task, ps::kServerGroup));
      Iterate(Workload::VAL);
      ++ cur_data_pass_;
    }

    if (is_predict) {
      printf("Prediction finished!\n");
      return true;
    }

    for (; cur_data_pass_ < max_data_pass_; ++cur_data_pass_) {
      if (Iterate(Workload::TRAIN) || Iterate(Workload::VAL)) {
        printf("Hit stop critera\n"); break;
      }
      if (cur_data_pass_ == max_data_pass_ - 1) {
        printf("Hit max number of data passes\n"); break;
      }
      SaveModel(false);
    }

    SaveModel(true);
    printf("Training finished!\n");
    return true;
  }

  virtual void ProcessResponse(ps::Message* response) {
    IterCmd cmd(response->task.cmd());
    if (cmd.process()) {
      auto id = response->sender;

      if (response->task.msg().size()) {
        CHECK(use_worker_local_data_);
        StringStream ss(response->task.msg());
        Workload wl; wl.Load(&ss);
        pool_.Add(wl.file, num_parts_per_file_, id);
        return;
      }

      pool_.Finish(id);
      Workload wl; pool_.Get(id, &wl);
      if (!wl.Empty()) {
        wl.type = cur_task_;
        wl.data_pass = cur_data_pass_;
        wl.file[0].format = data_format_;
        SendWorkload(id, wl);
      }
    }
  }

 private:

  /// \brief one iteration. returns true if time for stop
  bool Iterate(Workload::Type type) {
    cur_task_ = type;
    bool stop = false;
    std::string data;
    bool is_train = type == Workload::TRAIN;
    bool is_predict = predict_out_.size();
    if (is_train) {
      data = train_data_;
      printf("Training #iter = %d\n", cur_data_pass_);
    } else {
      data = val_data_;
      if (is_predict) {
        printf("Predicting\n");
      } else {
        printf("Validating #iter = %d\n", cur_data_pass_);
        if (data.empty()) return stop;
      }
    }
    if (data.empty()) fprintf(stderr, "WARNING: empty data\n");

    pool_.Clear(); pool_.Init(shuffle_, straggler_);

    if (use_worker_local_data_) {
      // ask the workers to match the files
      Workload wl;
      wl.file.resize(1);
      wl.file[0].filename = data;
      wl.file[0].n = 0;
      Wait(SendWorkload(ps::kWorkerGroup, wl));
    } else {
      // i will do it
      Workload wl; pool_.Match(data, &wl);
      pool_.Add(wl.file, num_parts_per_file_);
      if (is_predict) {
        CHECK_EQ(wl.file.size(), (size_t)1)
            << "use single file for prediction";
      }
      int npart = wl.file.size() * num_parts_per_file_;
      if (cur_data_pass_ == 0 && (npart < ps::NodeInfo::NumWorkers())) {
        fprintf(stderr, "WARNING: # of data parts (%d) < # of workers (%d)\n",
                npart, ps::NodeInfo::NumWorkers());
        fprintf(stderr, "         You may want to increase \"num_parts_per_file\"\n");
      }
    }

    // ask all workers to start by sending an empty workload
    Workload wl; SendWorkload(ps::kWorkerGroup, wl);

    // print every k sec for training
    auto disp = ProgHeader();
    if (disp.size()) {
      printf("  sec %s\n", disp.c_str());
      fflush(stdout);
    }

    while (!pool_.IsFinished()) {
      sleep(print_sec_);
      if (is_train) {
        if (ShowProgress(is_train)) {
          stop = true;
          pool_.ClearRemain();
        }
      }
    }

    // print progress for validation
    if (!is_train) {
      stop = ShowProgress(is_train);
    }
    return stop;
  }

  // return true if it's time for stopping
  bool ShowProgress(bool is_train) {
    Progress prog; monitor_.Get(&prog);
    auto disp = ProgString(prog);
    if (disp.empty()) return false;  // no progress...

    printf("%5.0lf  %s\n", GetTime() - start_time_, disp.c_str());
    fflush(stdout);

    return Stop(prog, is_train);
  }

  int SendWorkload(const std::string id, const Workload& wl) {
    StringStream ss; wl.Save(&ss);
    ps::Task task; task.set_msg(ss.str());
    IterCmd cmd; cmd.set_process();
    task.set_cmd(cmd.cmd); return Submit(task, id);
  }

  void SaveModel(bool force) {
    if (model_out_.size() == 0) return;
    if (force || (save_iter_ > 0 && (cur_data_pass_+1) % save_iter_ == 0)) {
      int iter = force ? 0 : cur_data_pass_;
      if (iter == 0) {
        printf("Saving final model to %s\n", model_out_.c_str());
      } else {
        printf("Saving model to %s-iter_%d\n", model_out_.c_str(), iter);
      }
      IterCmd cmd; cmd.set_save_model(); cmd.set_iter(iter);
      ps::Task task; task.set_cmd(cmd.cmd); task.set_msg(model_out_);
      Wait(Submit(task, ps::kServerGroup));
    }
  }

  ps::Root<double> monitor_;
  WorkloadPool pool_;
  double start_time_;
};


/**
 * \brief A server node. One must implement \ref SaveModel and \ref LoadModel
 */
class IterServer : public ps::App {
 protected:

  /// \brief Save model to disk
  virtual void SaveModel(Stream* fo) const = 0;

  /// \brief Load model from disk
  virtual void LoadModel(Stream* fi) = 0;

  /// \brief Report the progress to the scheduler
  void Report(const std::vector<double>& prog) { reporter_.Push(prog); }

  // implementation
 public:
  IterServer() {}
  virtual ~IterServer() {}

  virtual void ProcessRequest(ps::Message* request) {
    IterCmd cmd(request->task.cmd());
    auto filename = ModelName(request->task.msg(), cmd.iter());
    if (cmd.save_model()) {
      Stream* fo = CHECK_NOTNULL(Stream::Create(filename.c_str(), "w"));
      SaveModel(fo);
    } else if (cmd.load_model()) {
      Stream* fi = CHECK_NOTNULL(Stream::Create(filename.c_str(), "r"));
      LoadModel(fi);
    }
  }
 private:
  std::string ModelName(const std::string& base, int iter) {
    CHECK(base.size()) << "empty model name";
    std::string name = base;
    if (iter > 0) name += "_iter-" + std::to_string(iter);
    return name + "_part-" + std::to_string(ps::NodeInfo::MyRank());
  }

 public:
  ps::Slave<double> reporter_;
};

/**
 * \brief A worker node. One must implement \ref Process
 */
class IterWorker : public ps::App {
 protected:
  /// \brief Process a workerload sent from the scheduler
  virtual void Process(const Workload& wl) = 0;

  /// \brief Report the progress to the scheduler
  void Report(const std::vector<double>& prog) { reporter_.Push(prog); }

  // implementation
 public:
  IterWorker() { }
  virtual ~IterWorker() { }

  virtual void ProcessRequest(ps::Message* request) {
    IterCmd cmd(request->task.cmd());
    if (cmd.process()) {
      StringStream ss(request->task.msg());
      Workload wl; wl.Load(&ss);

      if (wl.Empty()) return;
      if (wl.file.size() == 1 && wl.file[0].n == 0) {
        // match my local files
        Workload local;
        WorkloadPool::Match(wl.file[0].filename, &local);
        StringStream ss; local.Save(&ss);
        ps::Task res; res.set_msg(ss.str());
        Reply(request, res);
      } else {
        // process
        Process(wl);
      }
    }
  }
 private:
  ps::Slave<double> reporter_;
};

}  // namespace solver
}  // namespace dmlc

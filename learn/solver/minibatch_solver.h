/**
 * @file   minibatch_solver.h
 * @brief  Template for an asynchronous minibatch solver
 */
#include "solver/iter_solver.h"
#include "base/minibatch_iter.h"
namespace dmlc {
namespace solver {

class MinibatchScheduler : public IterScheduler {
 protected:
  /// \brief training data, supports regular expression
  std::string train_data_;

  /// \brief validation data, supports regular expression
  std::string val_data_;

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
  virtual std::string ProgHeader() { return ""; }

  /**
   * \brief returns the string for the progress. empty string means no progress
   */
  virtual std::string ProgString(const Progress& prog) { return ""; }

  /**
   * \brief Init by a protobuf configure
   */
  template <typename Config>
  void Init(const Config& conf) {
    train_data_            = conf.train_data();
    val_data_              = conf.val_data();
    data_format_           = conf.data_format();
    num_parts_per_file_    = conf.num_parts_per_file();
    use_worker_local_data_ = conf.local_data();
    max_data_pass_         = conf.max_data_pass();
    print_sec_             = conf.print_sec();
    save_iter_             = conf.save_iter();
    load_iter_             = conf.load_iter();
    model_in_              = conf.model_in();
    model_out_             = conf.model_out();
    predict_out_           = conf.predict_out();
  }

 public:
  MinibatchScheduler() { }
  virtual ~MinibatchScheduler() { }

  /// \brief run iterations
  virtual bool Run() {
    printf("Connected %d servers and %d workers\n",
           ps::NodeInfo::NumServers(), ps::NodeInfo::NumWorkers());

    start_time_ = GetTime();

    bool is_predict = predict_out_.size();
    if (is_predict) {
      CHECK(model_in_.size()) << "should provide model_in for predicting";
    }

    int cur_iter = 0;
    if (model_in_.size()) {
      if (load_iter_ > 0) {
        printf("Loading model from iter = %d\n", load_iter_);
        cur_iter = load_iter_;
      } else {
        printf("Loading the last model\n");
        cur_iter = -1;
      }
      Wait(LoadModel(model_in_, cur_iter));
      ++ cur_iter;
    }

    if (is_predict) {
      Iterate(cur_iter, Workload::PRED);
      printf("Prediction is finished!\n");
      return true;
    }
    for (; cur_iter < max_data_pass_; ++cur_iter) {
      if (Iterate(cur_iter, Workload::TRAIN) || Iterate(cur_iter, Workload::VAL)) {
        printf("Hit stop critera\n"); break;
      }
      if (cur_iter == max_data_pass_ - 1) {
        printf("Hit max number of data passes %d\n", max_data_pass_);
        break;
      }
      if (model_out_.size() && save_iter_ > 0 && (cur_iter+1) % save_iter_ == 0) {
        printf("Saving model for iter = %d\n", cur_iter);
        Wait(SaveModel(model_out_, cur_iter));
      }
    }

    if (model_out_.size()) {
      printf("Saving the final model\n");
      Wait(SaveModel(model_out_, -1));
    }
    printf("Training is finished!\n");
    return true;
  }
 private:
  /// \brief one iteration. returns true if time for stop
  bool Iterate(int iter, Workload::Type type) {
    bool is_train = type == Workload::TRAIN;

    if (is_train) {
      data_filename_ = train_data_;
      printf("Training: iter = %d\n", iter);
    } else {
      data_filename_ = val_data_;
      if (type == Workload::PRED) {
        printf("Predicting\n");
      } else {
        printf("Validating: iter = %d\n", iter);
        if (data_filename_.empty()) return false;
      }
    }

    workload_.data_pass = iter;
    workload_.type      = type;


    auto disp = ProgHeader();
    if (disp.size()) {
      printf("  sec %s\n", disp.c_str());
      fflush(stdout);
    }

    bool stop = false;
    StartDispatch();

    // print every k sec for training
    while (!IsFinished()) {
      sleep(print_sec_);
      if (is_train) {
        stop = ShowProgress(is_train);
        if (stop) StopDispatch();  // wait all assigned workload finished
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
    auto prog = GetProgress();
    auto disp = ProgString(prog);
    if (disp.empty()) return false;  // no progress...
    printf("%5.0lf  %s\n", GetTime() - start_time_, disp.c_str());
    fflush(stdout);
    return Stop(prog, is_train);
  }

  double start_time_;
};


using MinibatchServer = IterServer;

class MinibatchWorker : public IterWorker {
 protected:
  /**
   * \brief feature ID type
   */
  using FeaID = ps::Key;

  /**
   * \brief minibatch, which contains a sparse matrix X with a vector Y
   */
  using Minibatch = dmlc::RowBlock<FeaID>;

  /**
   * \brief minibatch size
   */
  int mb_size_ = 10000;

  /**
   * \brief maximal concurrent minibatches being processing at the same time
   */
  int concurrent_mb_ = 1;

  /**
   * \brief If > 0, then the minibatch is randomly selected among \a mb_size_ *
   * \a shuffle_ examples.
   */
  int shuffle_ = 0;

  /**
   * \brief randomly down sampling negative examples
   */
  float neg_sampling_ = 1.0;

  /**
   * \brief minibatch size for validation or predicting
   */
  int val_mb_size_ = 10000000;

  /**
   * \brief maximal concurrent minibatches being processing at the same time for
   * validation or predicting
   */
  int val_concurrent_mb_ = 10;

  /**
   * \brief the time spent on real workload such as computing gradients. for
   * profiling usage
   */
  double workload_time_ = 0;

  /**
   * \brief Process one minibatch
   */
  virtual void ProcessMinibatch(const Minibatch& mb, const Workload& wl) = 0;

  /**
   * \brief Mark one minibatch is finished
   *
   * one must call this function when the minibatch is actually done, namely the
   * gradients have been pushed to the servers nodes
   */
  void FinishMinibatch() {
    // wake the main thread
    mb_mu_.lock(); -- num_mb_fly_; ++ num_mb_done_; mb_mu_.unlock();
    mb_cond_.notify_one();

    // log info
    double time = (GetTime() - start_);
    std::string overhead;
    if (workload_time_ > 0) {
      overhead = "overhead " + std::to_string(
          std::max(time - workload_time_, (double)0) / time * 100) + "%, ";
    }
    LOG(INFO) << num_mb_done_ << " done, avg time "
              << time / num_mb_done_ << ", " << overhead
              << num_mb_fly_ << " on running";
  }

  // implementation
 public:
  MinibatchWorker() { }
  virtual ~MinibatchWorker() { }

 protected:
  virtual void Process(const Workload& wl) {
    bool  train   = wl.type == Workload::TRAIN;
    int   mb_size = train ? mb_size_ : val_mb_size_;
    int   shuffle = train ? mb_size_ * shuffle_ : 0;
    float neg_sp  = train ? neg_sampling_ : 1.0;
    int max_mb    = wl.type == Workload::PRED ? 1 :
                    (train ? concurrent_mb_ : val_concurrent_mb_);
    LOG(INFO) << wl.ShortDebugString()
              << ", minibatch = " << mb_size
              << ", concurrency = " <<  max_mb
              << ", shuffle ratio = " << shuffle
              << ", negative sampling = " << neg_sp;

    num_mb_fly_ = num_mb_done_ = 0;
    start_ = GetTime();
    workload_time_ = 0;

    CHECK_GE(wl.file.size(), (size_t)1);
    auto file = wl.file[0];
    dmlc::data::MinibatchIter<FeaID> reader(
        file.filename.c_str(), file.k, file.n, file.format.c_str(),
        mb_size, shuffle, neg_sp);
    reader.BeforeFirst();
    while (reader.Next()) {
      WaitMinibatch(max_mb);
      ProcessMinibatch(reader.Value(), wl);
      mb_mu_.lock(); ++ num_mb_fly_; mb_mu_.unlock();
    }

    // wait untill all are done
    WaitMinibatch(1);
  }

 private:
  // wait until the currenta number of on processing minibatch < num
  inline void WaitMinibatch(int num) {
    std::unique_lock<std::mutex> lk(mb_mu_);
    mb_cond_.wait(lk, [this, num] {return num_mb_fly_ < num;});
  }

  int num_mb_fly_;
  int num_mb_done_;
  std::mutex mb_mu_;
  std::condition_variable mb_cond_;
  double start_;

};

}  // namespace solver
}  // namespace dmlc

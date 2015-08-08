/**
 * @file   iter_solver.h
 * @brief  Template for an iterate solver
 */
#include "base/progress.h"
#include "base/progress.h"
#include "ps.h"
namespace dmlc {
namespace wormhole {

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
  /// * batch model. the scheduler gives a worker \a n / \a num_workers parts
  /// each time. This assignment does not change between iterations
  ///
  /// * online model. The scheduler gives one part to a worker each time. The
  /// assignment is random, which is depended on the speeds of each worker
  bool batch_ = false;

  /// \brief whether shuffle the data parts before assigning
  bool shuffle_ = false;

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

  /// \brief training task or prediction task
  bool train_ = true;

  /**
   * \brief a user-defined stop criteria. stop the system if returns true
   *
   * @param prog the current progress
   * @param train in training or validation
   */
  virtual bool Stop(const Progress& prog, bool train) {
    return false;
  }

 public:
  // implementation TODO

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
  void Report(const Progress* const prog) {

  }

 public:
  // inplementation

};

/**
 * \brief A worker node. One must implement \ref Process
 */
class IterWorker : public ps::App {
 protected:
  /// \brief Process a workerload sent from the scheduler
  virtual void Process(const Workload& wl) = 0;

  /// \brief Report the progress to the scheduler
  void Report(const Progress* const prog) {

  }

 public:
  // inplementation TODO
};

}  // namespace wormhole
}  // namespace dmlc

/**
 * @file   data_parallel.h
 * @brief  Template for the scheduler dispatches data into worker nodes
 */
#include "ps.h"
#include "base/string_stream.h"
#include "base/workload.h"
#include "base/workload_pool.h"

namespace dmlc {
namespace solver {

/**
 * \brief encode/decode a command
 */
struct DataParCmd {
  DataParCmd() {}
  DataParCmd(int c) : cmd(c) {}

  // mutators
  void set_process() { cmd |= 1; }

  // accessors
  bool process() const { return cmd & 1; }

  int cmd = 0;
};

/**
 * \brief the scheduler node for dispatching data
 */
class DataParScheduler : public ps::App {
 protected:
  /**
   * \brief supports regular expression such as part-[0-9].*
   */
  std::string data_filename_;

  /**
   * \brief libsvm, crb, ...
   */
  std::string data_format_;

  /**
   * \brief  virtually partition a file into n parts for better loadbalance.
   */
  int num_parts_per_file_ = 10;

  /**
   * \brief batch or online assignment
   *
   * assume there are n parts ( = num_files_matched * num_parts_per_file_ )
   *
   * * batch mode. the scheduler gives a worker \a n / \a num_workers parts
   * each time. This assignment does not change between iterations
   *
   * * online mode. The scheduler gives one part to a worker each time. The
   * assignment is random, which is depended on the speeds of each worker
   */
  bool batch_ = false;

  /**
   * \brief whether shuffle the data parts before assigning
   */
  bool shuffle_ = true;

  /**
   * \brief mark a worker as the straggler if it uses \a straggler_ times
   * longer than average. on the online mode, the scheduler will reassign that
   * workerload to another worker.
   */
  int straggler_ = 3;

  /**
   * \brief whether allow each worker to tell the scheduler which file it can
   * access
   *
   * If true, each worker matches the files it can access and then report to
   * the scheduler. It is useful when the data have been dispatched into
   * workers' local disks. it can reduce the cost to access data remotely
   */
  bool use_worker_local_data_ = false;

  /**
   * \brief the base workload, it can contains info an app want to send to the workers
   */

  Workload workload_;

  /**
   * \brief Start dispatching
   */
  void StartDispatch() {
    pool_.Clear(); pool_.Init(shuffle_, straggler_);

    if (use_worker_local_data_) {
      // ask the workers to match the files
      Workload wl; wl.file.resize(1);
      wl.file[0].filename = data_filename_; wl.file[0].n = 0;
      Wait(SendWorkload(ps::kWorkerGroup, wl));
    } else {
      // let me match the files
      Workload wl; pool_.Match(data_filename_, &wl);
      pool_.Add(wl.file, num_parts_per_file_);
      int npart = wl.file.size() * num_parts_per_file_;
      if (npart < ps::NodeInfo::NumWorkers()) {
        fprintf(stderr, "WARNING: # of data parts (%d) < # of workers (%d)\n",
                npart, ps::NodeInfo::NumWorkers());
        fprintf(stderr, "         You may increase \"num_parts_per_file\"\n");
      }
    }

    // ask all workers to start by sending an empty workload
    Workload wl; SendWorkload(ps::kWorkerGroup, wl);
  }

  /**
   * \brief Query if dispatching is finished, namely the workers have processed
   * all data specified by \a data_filename_
   */
  bool IsFinished() { return pool_.IsFinished(); }

  /**
   * \brief Stop to assign new workload to workers. But cannot revoked the ones
   * already assigned
   */
  void StopDispatch() { pool_.ClearRemain(); }

  // implementation
 public:
  DataParScheduler() {
    sys_.manager().AddNodeFailureHandler([this](const std::string& id) {
        pool_.Reset(id);
      });
  }
  virtual ~DataParScheduler() { }

  virtual void ProcessResponse(ps::Message* response) {
    DataParCmd cmd(response->task.cmd());
    if (!cmd.process()) return;
    auto id = response->sender;

    // add workers' locally matched files to the pool
    if (response->task.msg().size()) {
      CHECK(use_worker_local_data_);
      StringStream ss(response->task.msg());
      Workload wl; wl.Load(&ss);
      pool_.Add(wl.file, num_parts_per_file_, id);
      return;
    }

    // a worker finished a workload, assign it a new one if available
    pool_.Finish(id);
    Workload wl = workload_; pool_.Get(id, &wl);
    if (wl.Empty()) return;
    wl.file[0].format = data_format_;
    SendWorkload(id, wl);
  }

 private:
  int SendWorkload(const std::string id, const Workload& wl) {
    StringStream ss; wl.Save(&ss);
    ps::Task task; task.set_msg(ss.str());
    DataParCmd cmd; cmd.set_process();
    task.set_cmd(cmd.cmd); return Submit(task, id);
  }

  WorkloadPool pool_;
};


/**
 * \brief The worker node for processing workload. One must implement \ref Process
 */
class DataParWorker : public ps::App {
 protected:
  /**
   * \brief Process a workerload sent from the scheduler
   */
  virtual void Process(const Workload& wl) = 0;

  // implementation
 public:
  DataParWorker() { }
  virtual ~DataParWorker() { }

  virtual void ProcessRequest(ps::Message* request) {
    DataParCmd cmd(request->task.cmd());
    if (cmd.process()) {
      StringStream ss(request->task.msg());
      Workload wl; wl.Load(&ss);

      if (wl.Empty()) return;
      if (wl.file.size() == 1 && wl.file[0].n == 0) {
        // match my local files
        Workload local; WorkloadPool::Match(wl.file[0].filename, &local);
        StringStream ss; local.Save(&ss);
        ps::Task res; res.set_msg(ss.str());
        Reply(request, res);
      } else {
        // process
        Process(wl);
      }
    }
  }
};

}  // namespace solver
}  // namespace dmlc

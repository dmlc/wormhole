/**
 * @file   iter_solver.h
 * @brief  Template for an iterate solver
 */
#include "solver/data_parallel.h"
namespace dmlc {
namespace solver {

using Progress = std::vector<double>;

/**
 * \brief encode/decode a command
 */
struct IterCmd : public DataParCmd {
  IterCmd() {}
  IterCmd(int c) : DataParCmd(c) {}

  // mutators
  void set_iter(int iter) { cmd += (iter+1) << 16; }
  void set_load_model() { cmd |= 1<<1; }
  void set_save_model() { cmd |= 1<<2; }

  // accessors
  bool load_model() const { return cmd & 1<<1; }
  bool save_model() const { return cmd & 1<<2; }
  bool iter() const { return (cmd >> 16)-1; }
};

/**
 * \brief the scheduler node for an iterate solver
 */
class IterScheduler : public DataParScheduler {
 protected:
  /**
   * \brief Ask all servers to load model, return the timestamp of this request
   *
   * @param filename model filename
   * @param iter load from a particualr iteration. if -1, then load the last
   */
  int LoadModel(const std::string& filename, int iter) {
    IterCmd cmd; cmd.set_load_model(); cmd.set_iter(iter);
    ps::Task task; task.set_cmd(cmd.cmd); task.set_msg(filename);
    return Submit(task, ps::kServerGroup);
  }

  /**
   * \brief Ask all servers to save model, return the timestamp of this request
   *
   * @param filename model filename
   * @param iter save for a particualr iteration. if -1, then saved as the last
   */
  int SaveModel(const std::string& filename, int iter) {
    IterCmd cmd; cmd.set_save_model(); cmd.set_iter(iter);
    ps::Task task; task.set_cmd(cmd.cmd); task.set_msg(filename);
    return Submit(task, ps::kServerGroup);
  }

  /**
   * \brief Returns the aggregated progress among all woreker/servers since the
   * last time calling this function
   */
  Progress GetProgress() { Progress prog; monitor_.Get(&prog); return prog; }

  // implementation
 public:
  IterScheduler() { }
  virtual ~IterScheduler() { }

 private:
  ps::Root<double> monitor_;
};


/**
 * \brief A server node. One must implement \ref SaveModel and \ref LoadModel
 */
class IterServer : public ps::App {
 protected:
  /**
   * \brief Save model to disk
   */
  virtual void SaveModel(Stream* fo) const = 0;

  /**
   * \brief Load model from disk
   */
  virtual void LoadModel(Stream* fi) = 0;

  /**
   * \brief Report the progress to the scheduler
   */
  void ReportToScheduler(const Progress& prog) { reporter_.Push(prog); }

  // implementation
 public:
  IterServer() {}
  virtual ~IterServer() {}

  virtual void ProcessRequest(ps::Message* request) {
    if (request->task.msg().size() == 0) return;
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
    std::string name = base;
    if (iter >= 0) name += "_iter-" + std::to_string(iter);
    return name + "_part-" + std::to_string(ps::NodeInfo::MyRank());
  }
  ps::Slave<double> reporter_;
};

/**
 * \brief A worker node.
 */
class IterWorker : public DataParWorker {
 protected:

  /**
   * \brief Report the progress to the scheduler
   */
  void ReportToScheduler(const Progress& prog) { reporter_.Push(prog); }

  // implementation
 public:
  IterWorker() { }
  virtual ~IterWorker() { }

 private:
  ps::Slave<double> reporter_;
};

}  // namespace solver
}  // namespace dmlc

/**
 * @file   dist_monitor.h
 * @brief  distributed monitor for ps
 */
#pragma once
#include "ps.h"
#include "ps/app.h"
#include "base/progress.h"
#include "base/string_stream.h"
#include <mutex>
#include <unordered_map>

namespace dmlc {

template <typename Progress>
class ProgressMonitor : public ps::Customer {
 public:
  ProgressMonitor(int id = ps::NextCustomerID()) : ps::Customer(id) {}
  virtual ~ProgressMonitor() { }

  // Get the merged progress on channel chl, return the number of unique senders
  int Get(Progress *prog, int chl = 0) {
    std::lock_guard<std::mutex> lk(mu_);
    prog->Merge(&prog_[chl]);
    return nodes_[chl].size();
  }

  void Clear(int chl = 0) {
    std::lock_guard<std::mutex> lk(mu_);
    prog_[chl].Clear();
    nodes_[chl].clear();
  }

  // implement system required functions
  virtual void ProcessRequest(ps::Message* request) {
    std::lock_guard<std::mutex> lk(mu_);
    StringStream stream(request->task.msg());
    Progress p; p.Load(&stream);
    int chl = request->task.key_channel();
    prog_[chl].Merge(&p);
    nodes_[chl].insert(request->sender);
  }

 private:
  std::mutex mu_;
  std::unordered_map<int, Progress> prog_;
  std::unordered_map<int, std::unordered_set<std::string>> nodes_;
  Progress progress_;
};

class ProgressReporter : public ps::Customer {
 public:
  ProgressReporter(int id = ps::NextCustomerID()) : ps::Customer(id) { }
  virtual ~ProgressReporter() { }

  void Report(const IProgress *const prog, int chl = 0) {
    StringStream stream; prog->Save(&stream);
    ps::Task report; report.set_msg(stream.str());
    report.set_key_channel(chl);
    Submit(report, ps::SchedulerID());
  }
};

}  // namespace dmlc

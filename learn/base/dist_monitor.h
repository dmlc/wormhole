/**
 * @file   dist_monitor.h
 * @brief  distributed monitor for ps
 */
#pragma once
#include "ps.h"
#include "ps/app.h"
#include "base/progress.h"

namespace dmlc {


template <typename Progress>
class ProgressMonitor : public ps::Customer {
  ProgressMonitor(int id = ps::NextCustomerID()) : ps::Customer(id) {}
  virtual ~ProgressMonitor() { }

  // Get the merged progress on channel chl, return the number of unique senders
  int Get(Progress *prog, int chl = 0) {

  }

  void Clear(int chl = 0) {

  }

};

class ProgressReporter : public ps::Customer {
 public:
  ProgressReporter(double report_itv = 1, int id = ps::NextCustomerID())
      : ps::Customer(id) {
    report_itv_ = report_itv;
  }
  virtual ~ProgressReporter() { }

  void set_report_itv(double sec) { report_itv_ = sec; }

  void Report(const IProgress *const prog, int chl = 0) {

  }

  void Flush() {

  }
 private:
  double report_itv_ = 1, last_report_ = 0;
};

}  // namespace dmlc

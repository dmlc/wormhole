#pragma once
#include "base/monitor.h"
#include "ps/monitor.h"
#include "dmlc/timer.h"
namespace dmlc {
namespace linear {

class TimeReporter {
 public:
  explicit TimeReporter(double report_itv)  {
    set_report_itv(report_itv);
  }
  TimeReporter() { }
  ~TimeReporter() { }

  void set_report_itv(double sec) { report_itv_ = sec; }

  void Report(int chl, Progress* prog) {
    double tv = GetTime();
    chl_ = chl;
    prog_.Merge(*prog);
    prog->Clear();
    if (tv - last_report_ > report_itv_ ) {
      last_report_ = tv;
      Flush();
    }
  }

  // sent immediatly
  void Flush() {
    sch_.Report(chl_, prog_); prog_.Clear();
  }
 private:
  Progress prog_;
  int chl_ = 0;
  double report_itv_ = 1, last_report_ = 0;
  ps::MonitorSlaver<Progress> sch_;
};

struct DistModelMonitor : public ModelMonitor {
 public:
  DistModelMonitor(double report_itv) : reporter(report_itv) { }
  virtual ~DistModelMonitor() { }

  virtual void Report() { reporter.Report(0, &prog); }

  TimeReporter reporter;
};


}  // namespace linear
}  // namespace dmlc

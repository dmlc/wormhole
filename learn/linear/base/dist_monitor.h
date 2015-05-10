#pragma once
#include "base/monitor.h"
#include "ps/monitor.h"
#include "dmlc/timer.h"
namespace dmlc {
namespace linear {

class TimeReporter {
 public:
  TimeReporter(double report_itv) : itv_(report_itv) { }
  ~TimeReporter() { }

  void Report(int chl, Progress* prog) {
    double tv = GetTime();
    if (tv - last_report_ > itv_ ) {
      last_report_ = tv;
      sch_.Report(chl, *prog);
      prog->Clear();
    }
  }
 private:
  double itv_, last_report_ = 0;
  ps::MonitorSlaver<Progress> sch_;
};

struct DistModelMonitor : public ModelMonitor {
 public:
  DistModelMonitor(double report_itv) : reporter(report_itv) { }
  virtual ~DistModelMonitor() { }

  void Report(int chl = 0) { reporter.Report(chl, &prog); }

  TimeReporter reporter;
};


}  // namespace linear
}  // namespace dmlc

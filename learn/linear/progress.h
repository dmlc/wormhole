#pragma once
#include <vector>
#include <string>
namespace dmlc {
namespace linear {


struct Progress {
  Progress() : data(6) { }

  static std::string HeadStr() {
    return "  ttl #ex   inc #ex    |w|_0       logloss  accuracy     AUC";
  }

  std::string PrintStr() {
    ttl_ex += new_ex();
    nnz_w += new_w();

    if (new_ex() == 0) return "";

    char buf[256];
    snprintf(buf, 256, "%8.3g  %8.3g  %11.6g  %8.6lf  %8.6lf  %8.6lf",
             ttl_ex, new_ex(), nnz_w, objv() / new_ex(),
             acc() / count(), auc() / count());
    return std::string(buf);
  }

  // mutator
  double& objv() { return data[0]; }
  double& acc() { return data[1]; }
  double& auc() { return data[2]; }

  double& count() { return data[3]; }
  double& new_ex() { return data[4]; }
  double& new_w() { return data[5]; }

  std::vector<double> data;
  double ttl_ex = 0, nnz_w = 0;

};

} // namespace linear
} // namespace dmlc

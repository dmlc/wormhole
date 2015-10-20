#pragma once
#include <vector>
#include <string>
namespace dmlc {
namespace linear {

struct Progress {
  Progress() : data(10) { }

  static std::string HeadStr() {
    return "  ttl #ex   inc #ex    |w|_0    logloss  accuracy  precison  recall  neg_precision  neg_recall  AUC";
  }

  std::string PrintStr() {
    ttl_ex += new_ex();
    nnz_w += new_w();

    if (new_ex() == 0) return "";

    char buf[256];
    snprintf(buf, 256,  "%8.3g  %8.3g  %8.6g  %8.6lf  %8.6lf   %8.6lf  %8.6lf  %8.6lf   %8.6lf   %8.6lf"
             ttl_ex, 
             new_ex(), 
             nnz_w, objv() / new_ex(),
             acc() / count(), 
             precision() / count(), 
             recall() / count(), 
             neg_precision() / count(), 
             neg_recall() / count(), 
             auc() / count());
    return std::string(buf);
  }

  // mutator
  double& objv() { return data[0]; }
  double& acc() { return data[1]; }
  double& auc() { return data[2]; }
  double& precision() { return data[6];}
  double& recall() { return data[7];}
  double& neg_precision() { return data[8];}
  double& neg_recall() { return data[9];}
  double& count() { return data[3]; }
  double& new_ex() { return data[4]; }
  double& new_w() { return data[5]; }

  std::vector<double> data;
  double ttl_ex = 0, nnz_w = 0;

};

} // namespace linear
} // namespace dmlc

#pragma once
#include <algorithm>
#include <dmlc/logging.h>
#include <dmlc/omp.h>
namespace dmlc {

template <typename V>
class BinClassEval {
 public:
  BinClassEval(const V* const label,
             const V* const predict,
             size_t n,
             int num_threads = 2)
      : label_(label), predict_(predict), size_(n), nt_(num_threads) { }
  ~BinClassEval() { }

  V AUC() {
    size_t n = size_;
    struct Entry { V label; V predict; };
    std::vector<Entry> buff(n);
    for (size_t i = 0; i < n; ++i) {
      buff[i].label = label_[i];
      buff[i].predict = predict_[i];
    }
    std::sort(buff.data(), buff.data()+n,  [](const Entry& a, const Entry&b) {
        return a.predict < b.predict; });
    V area = 0, cum_tp = 0;
    for (size_t i = 0; i < n; ++i) {
      if (buff[i].label > 0) {
        cum_tp += 1;
      } else {
        area += cum_tp;
      }
    }
    if (cum_tp == 0 || cum_tp == n) return 1;
    area /= cum_tp * (n - cum_tp);
    return area < 0.5 ? 1 - area : area;
  }

  V Accuracy(V threshold) {
    V correct = 0;
    size_t n = size_;
#pragma omp parallel for reduction(+:correct) num_threads(nt_)
    for (size_t i = 0; i < n; ++i) {
      if ((label_[i] > 0 && predict_[i] > threshold) ||
          (label_[i] <= 0 && predict_[i] <= threshold))
        correct += 1;
    }
    V acc = correct / (V) n;
    return acc > 0.5 ? acc : 1 - acc;
  }
  V Precision(V threshold) {
    size_t n = size_;
    size_t i;
    V tp, fp; 
    V precision;
    tp = fp = 0;
    for (i = 0; i < n; ++i) {
        if (predict_[i] > threshold) {
            if (label_[i] > 0) {
                ++tp;
            } else {
                ++fp;
            }   
        }   
    }   
    if (tp + fp == 0) {
        precision = 0;
    } else {
        precision = tp / (V) (tp + fp);
    }   
    return precision;
  }
  V NegPrecision(V threshold) {
    size_t n = size_;
    size_t i;
    V tn, fn; 
    V neg_precision;
    tn = fn = 0;
    for (i = 0; i < n; ++i) {
        if (predict_[i] <= threshold) {
            if (label_[i] <= 0) {
                ++tn;
            } else {
                ++fn;
            }   
        }   
    }
    if (tn + fn == 0) {
        neg_precision = 0;
    } else {
        neg_precision = tn / (V) (tn + fn);
    }
    return neg_precision;
  }
   // recall 
  V Recall(V threshold) {
      size_t n = size_;
      size_t i;
      V tp, fn;
      V recall;
      tp = fn = 0;
      for (i = 0; i < n; ++i) {
          if (label_[i] > 0) {
              if (predict_[i] > threshold) {
                  ++tp;
              } else {
                  ++fn;
              }
          }
      }
      if (tp + fn == 0) {
          recall = 0;
      } else {
          recall = tp / (V) (tp + fn);
      }
      return recall;
  }
  V NegRecall(V threshold) {
      size_t n = size_;
      size_t i;
      V tn, fp;
      V neg_recall;
      tn = fp = 0;
      for (i = 0; i < n; ++i) {
          if (label_[i] <= 0) {
              if (predict_[i] <= threshold) {
                  ++tn;
              } else {
                  ++fp;
              }
          }
      }
      if (tn + fp == 0) {
          neg_recall = 0;
      } else {
          neg_recall = tn / (V) (tn + fp);
      }
      return neg_recall;
  }
  V LogLoss() {
    V loss = 0;
    size_t n = size_;
#pragma omp parallel for reduction(+:loss) num_threads(nt_)
    for (size_t i = 0; i < n; ++i) {
      V y = label_[i] > 0;
      V p = 1 / (1 + exp(- predict_[i]));
      p = p < 1e-10 ? 1e-10 : p;
      loss += y * log(p) + (1 - y) * log(1 - p);
    }
    return - loss;
  }

  V LogitObjv() {
    V objv = 0;
#pragma omp parallel for reduction(+:objv) num_threads(nt_)
    for (size_t i = 0; i < size_; ++i) {
      V y = label_[i] > 0 ? 1 : -1;
      objv += log( 1 + exp( - y * predict_[i] ));
    }
    return objv;
  }
  
  V Copc(){
    V clk = 0;
    V clk_exp = 0.0;
#pragma omp parallel for reduction(+:clk,clk_exp) num_threads(nt_)
    for (size_t i = 0; i < size_; ++i) {
      if (label_[i] > 0) clk += 1;
      clk_exp += 1.0 / ( 1.0 + exp( - predict_[i] ));
    }
    return clk / clk_exp;
  }

 private:
  V const* label_;
  V const* predict_;
  size_t size_;
  int nt_;
};


}  // namespace dmlc

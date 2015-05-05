#pragma once
#include <algorithm>
#include <sort>
#include <dmlc/logging.h>
namespace dmlc {
namespace linear {

class Evaluation {
 public:
  template <typename V>
  static V AUC(const std::vector<V>& label, const std::vector<V>& predict) {
    CHECK_EQ(label.size(), predict.size());
    return AUC(label.data(), predict.data(), label.size());
  }

  template <typename V>
  static V Accuracy(const std::vector<V>& label, const std::vector<V>& predict,
                    V threshold = 0) {
    CHECK_EQ(label.size(), predict.size());
    return Accuracy(label.data(), predict.data(), label.size(), threshold);
  }

  template <typename V>
  static V LogLoss(const std::vector<V>& label, const std::vector<V>& predict) {
    CHECK_EQ(label.size(), predict.size());
    return LogLoss(label.data(), predict.data(), label.size());
  }
};

template <typename V>
V Evaluation::AUC(const V* const label, const V* const predict, size_t n) {
  struct Entry {
    V label;
    V predict;
  };
  std::vector<Entry> buff(n);
  for (int i = 0; i < n; ++i) {
    buff[i].label = label[i];
    buff[i].predict = predict[i];
  }
  std::sort(buff.data(), buff.data()+n,  [](const Entry& a, const Entry&b) {
      return a.predict < b.predict; });
  V area = 0, cum_tp = 0;
  for (int i = 0; i < n; ++i) {
    if (buff[i].label > 0) {
      cum_tp += 1;
    } else {
      area += cum_tp;
    }
  }
  area /= cum_tp * (n - cum_tp);
  return area < 0.5 ? 1 - area : area;
}


template <typename V>
V Evaluation::Accuracy(const V* const label, const V* const predict, size_t n,
    V threshold) {
  V correct = 0;
  for (int i = 0; i < n; ++i) {
    if ((label[i] > 0 && predict[i] > threshold) ||
        (label[i] <= 0 && predict[i] <= threshold))
      correct += 1;
  }
  V acc = correct / (V) n;
  return acc > 0.5 ? acc : 1 - acc;
}


template <typename V>
V Evaluation::LogLoss(const V* const label, const V* const predict, size_t n) {
  V loss = 0;
  for (int i = 0; i < n; ++i) {
    V y = label[i] > 0;
    V p = 1 / (1 + exp(- predict[i]));
    loss += y * log(p) + (1 - y) * log(1 - p);
  }
  return - loss / n;
}

}  // namespace linear
}  // namespace dmlc

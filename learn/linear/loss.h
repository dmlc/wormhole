#pragma once
#include <dmlc/data.h>
#include "proto/linear.pb.h"

namespace dmlc {
namespace linear {

template<typename T> class Loss {
 public:
  using Data = RowBlock<unsigned>;
  using Param = std::vector<T>;
  // predict
  virtual void Pred(const Data& data, const Param& w) = 0;

  // evaluate the loss value
  virtual T Eval(const Data& data, const Param& w) = 0;

  // compute the gradients
  virtual void CalcGrad(const Data& data, const Param& w, Param* grad) = 0;

  // clear the temp results
  virtual void Clear() = 0;
};

// scalar loss, that is, a loss which takes as input a real value prediction and
// a real valued label and outputs a non-negative loss value. Examples include
// the hinge hinge loss, binary classification loss, and univariate regression
// loss.
template <typename T>
class ScalarLoss : public Loss<T> {
 public:
 protected:
  std::vector<T> Xw;  // X * w
};

template <typename T>
class LogitLoss : public ScalarLoss<T> {

};

template <typename T>
class SquareHingeLoss : public ScalarLoss<T> {

};

template<typename T>
static Loss<T>* CreateLoss(Config::Loss loss) {
  switch (loss) {
    case Config::LOGIT:
      return new LogitLoss<T>();
    case Config::SQUARE_HINGE:
      return new SquareHingeLoss<T>();
    default:
      LOG(FATAL) << "unknown type: " << loss;
  }
  return NULL;
}

}  // namespace linear
}  // namespace dmlc

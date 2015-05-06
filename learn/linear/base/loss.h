#pragma once
#include <dmlc/data.h>
#include <math.h>
#include "proto/linear.pb.h"
#include "base/spmv.h"
#include "base/eval_model.h"

namespace dmlc {
namespace linear {

/**
 * \brief Scalar loss
 *
 * a loss which takes as input a real value prediction and a
 * real valued label and outputs a non-negative loss value. Examples include the
 * hinge hinge loss, binary classification loss, and univariate regression loss.
 */
template<typename T>
class ScalarLoss {
 public:
  ScalarLoss() : init_(false), nthreads_(2) { }
  virtual ~ScalarLoss() { }

  using Data = RowBlock<unsigned>;

  void Init(const Data& data, const std::vector<T>& w) {
    data_ = data;
    Xw_.resize(data_.size);
    SpMV::Times(data_, w, &Xw_, nthreads_);
    init_ = true;
  }

  /*! \brief evaluate the loss value */
  virtual T Objv() = 0;

  /*! \brief compute the gradients */
  virtual void CalcGrad(std::vector<T>* grad) = 0;

  T AUC() {
    CHECK(init_);
    return Evaluation::AUC(data_.label, Xw_.data(), Xw_.size());
  }

  T Accuracy(T threshold = 0) {
    CHECK(init_);
    return Evaluation::Accuracy(data_.label, Xw_.data(), Xw_.size(), threshold);
  }

  T LogLoss() {
    CHECK(init_);
    return Evaluation::LogLoss(data_.label, Xw_.data(), Xw_.size());
  }
 protected:
  bool init_;
  Data data_;
  std::vector<T> Xw_;  // X * w
  int nthreads_;
};

template <typename T>
class LogitLoss : public ScalarLoss<T> {
 public:
  using ScalarLoss<T>::data_;
  using ScalarLoss<T>::Xw_;
  using ScalarLoss<T>::nthreads_;
  virtual T Objv() {
    CHECK(this->init_);
    T ret = 0;

#pragma omp parallel for reduction(+:ret) num_threads(nthreads_)
    for (size_t i = 0; i < data_.size; ++i) {
      T y = data_.label[i] > 0 ? 1 : -1;
      ret += log( 1 + exp( - y * Xw_[i] ));
    }
    return ret;
  }

  virtual void CalcGrad(std::vector<T>* grad) {
    CHECK(this->init_);
    std::vector<T> dual(data_.size);
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < data_.size; ++i) {
      T y = data_.label[i] > 0 ? 1 : -1;
      dual[i] = - y / ( 1 + exp ( y * Xw_[i] ));
    }
    SpMV::TransTimes(data_, dual, grad, nthreads_);
  }
};

template <typename T>
class SquareHingeLoss : public ScalarLoss<T> {
 public:
  using ScalarLoss<T>::data_;
  using ScalarLoss<T>::Xw_;
  using ScalarLoss<T>::nthreads_;
  virtual T Objv() {
    CHECK(this->init_);
    T ret = 0;
#pragma omp parallel for reduction(+:ret) num_threads(nthreads_)
    for (size_t i = 0; i < data_.size; ++i) {
      T y = data_.label[i] > 0 ? 1 : -1;
      T tmp = std::max(1 - y * Xw_[i], (T)0);
      ret += tmp * tmp;
    }
    return ret;
  }

  virtual void CalcGrad(std::vector<T>* grad) {
    CHECK(this->init_);

    std::vector<T> dual(data_.size);
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < data_.size; ++i) {
      T y = data_.label[i] > 0 ? 1 : -1;
      dual[i] = y * (y * Xw_[i] > 1.0);
    }
    SpMV::TransTimes(data_, dual, grad, nthreads_);

#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < grad->size(); ++i) {
      (*grad)[i] *= -2.0;
    }
  }
};

template<typename T>
static ScalarLoss<T>* CreateLoss(Config::Loss loss) {
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

#pragma once
#include <dmlc/data.h>
#include <dmlc/io.h>
#include <math.h>
#include "config.pb.h"
#include "progress.h"
#include "base/spmv.h"
#include "base/binary_class_evaluation.h"
namespace dmlc {
namespace linear {

/**
 * \brief Scalar loss
 *
 * a loss which takes as input a real value prediction and a
 * real valued label and outputs a non-negative loss value. Examples include the
 * hinge hinge loss, binary classification loss, and univariate regression loss.
 */
template <typename V>
class ScalarLoss {
 public:
  ScalarLoss() : init_(false) { }
  virtual ~ScalarLoss() { }

  /**
   * \brief init
   *
   * @param data X and Y
   * @param w weight
   * @param nt num of threads
   */
  void Init(const RowBlock<unsigned>& data,
            const std::vector<V>& w, int nt) {
    data_ = data;
    Xw_.resize(data_.size);
    SpMV::Times(data_, w, &Xw_, nt_);
    nt_ = nt;
    init_ = true;
  }

  /*! \brief evaluate the loss value */
  virtual void Evaluate(Progress* prog) {
    CHECK(init_);
    prog->new_ex()  = data_.size;
    prog->count()   = 1;
  }

  /*! \brief compute the gradients */
  virtual void CalcGrad(std::vector<V>* grad) = 0;

  /**
   * \brief save prediction
   * \param prob_out output probability
   */
  virtual void Predict(Stream* fo, bool prob_out) {
    CHECK(init_); CHECK_NOTNULL(fo);
    ostream os(fo);
    if (prob_out) {
      for (auto p : Xw_) os << 1.0 / (1.0 + exp( - p )) << "\n";
    } else {
      for (auto p : Xw_) os << p << "\n";
    }
  }

 protected:
  bool init_;
  RowBlock<unsigned> data_;
  std::vector<V> Xw_;  // X * w
  int nt_;
};

/**
 * \brief binary classification with label y = +1 / -1
 */
template <typename V>
class BinClassLoss : public ScalarLoss<V> {
 public:
  using ScalarLoss<V>::data_;
  using ScalarLoss<V>::Xw_;
  using ScalarLoss<V>::nt_;
  virtual void Evaluate(Progress* prog) {
    ScalarLoss<V>::Evaluate(prog);
    BinClassEval<V> eval(data_.label, Xw_.data(), Xw_.size(), nt_);
    prog->auc()     = eval.AUC();
    prog->acc()     = eval.Accuracy(0);
  }
};

/**
 * \brief logistic loss: \f$ log(1+exp(−y \langle x, w \rangle)) \f$
 */
template <typename V>
class LogitLoss : public BinClassLoss<V> {
 public:
  using ScalarLoss<V>::data_;
  using ScalarLoss<V>::Xw_;
  using ScalarLoss<V>::nt_;
  using ScalarLoss<V>::init_;

  virtual void Evaluate(Progress* prog) {
    BinClassLoss<V>::Evaluate(prog);
    BinClassEval<V> eval(data_.label, Xw_.data(), Xw_.size(), nt_);
    prog->objv() = eval.LogitObjv();
  }

  virtual void CalcGrad(std::vector<V>* grad) {
    CHECK(init_);
    std::vector<V> dual(data_.size);
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < data_.size; ++i) {
      V y = data_.label[i] > 0 ? 1 : -1;
      dual[i] = - y / ( 1 + exp ( y * Xw_[i] ));
    }
    SpMV::TransTimes(data_, dual, grad, nt_);
  }
};

/**
 * \brief square hinge loss: \f$ \max\left(0, (1-yp)^2\right) \f$
 */
template <typename V>
class SquareHingeLoss : public BinClassLoss<V> {
 public:
  using ScalarLoss<V>::data_;
  using ScalarLoss<V>::Xw_;
  using ScalarLoss<V>::nt_;
  using ScalarLoss<V>::init_;

  virtual void Evaluate(Progress* prog) {
    BinClassLoss<V>::Evaluate(prog);
    V objv = 0;
#pragma omp parallel for reduction(+:objv) num_threads(nt_)
    for (size_t i = 0; i < data_.size; ++i) {
      V y = data_.label[i] > 0 ? 1 : -1;
      V tmp = std::max(1 - y * Xw_[i], (V)0);
      objv += tmp * tmp;
    }
    prog->objv() = objv;
  }

  virtual void CalcGrad(std::vector<V>* grad) {
    CHECK(init_);

    std::vector<V> dual(data_.size);
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < data_.size; ++i) {
      V y = data_.label[i] > 0 ? 1 : -1;
      dual[i] = y * (y * Xw_[i] > 1.0);
    }
    SpMV::TransTimes(data_, dual, grad, nt_);

#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < grad->size(); ++i) {
      (*grad)[i] *= -2.0;
    }
  }
};

/**
 * \brief squared loss \f$ \frac12 (p-y)^2 \f$
 */
template <typename V>
class SquareLoss : public ScalarLoss<V> {
 public:
  // TODO
};

/**
 * \brief loss factory
 */
template <typename V>
static ScalarLoss<V>* CreateLoss(Config::Loss loss) {
  switch (loss) {
    case Config::LOGIT:
      return new LogitLoss<V>();
    case Config::SQUARE_HINGE:
      return new SquareHingeLoss<V>();
    default:
      LOG(FATAL) << "unknown type: " << loss;
  }
  return NULL;
}

}  // namespace linear
}  // namespace dmlc

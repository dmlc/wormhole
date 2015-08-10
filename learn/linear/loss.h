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
template <typename Real>
class ScalarLoss {
 public:
  ScalarLoss() : init_(false) { }
  virtual ~ScalarLoss() { }

  void Init(const RowBlock<unsigned>& data,
            const std::vector<Real>& w, int nt) {
    data_ = data;
    Xw_.resize(data_.size);
    SpMV::Times(data_, w, &Xw_, nt_);
    nt_ = nt;
    init_ = true;
  }

  /*! \brief evaluate the loss value */
  virtual void Evaluate(Progress* prog) {
    CHECK(init_);
    BinClassEval<Real> eval(data_.label, Xw_.data(), Xw_.size(), nt_);
    prog->auc()     = eval.AUC();
    prog->acc()     = eval.Accuracy(0);
    prog->new_ex()  = data_.size;
    prog->count()   = 1;
  }

  /*! \brief compute the gradients */
  virtual void CalcGrad(std::vector<Real>* grad) = 0;

  void SavePrediction(Stream* fo) {
    CHECK(init_);
    CHECK_NOTNULL(fo);
    ostream os(fo);
    for (auto p : Xw_) os << p << "\n";
  }

 protected:
  bool init_;
  RowBlock<unsigned> data_;
  std::vector<Real> Xw_;  // X * w
  int nt_;
};

template <typename Real>
class LogitLoss : public ScalarLoss<Real> {
 public:
  using ScalarLoss<Real>::data_;
  using ScalarLoss<Real>::Xw_;
  using ScalarLoss<Real>::nt_;
  using ScalarLoss<Real>::init_;

  virtual void Evaluate(Progress* prog) {
    ScalarLoss<Real>::Evaluate(prog);
    BinClassEval<Real> eval(data_.label, Xw_.data(), Xw_.size(), nt_);
    prog->objv() = eval.LogitObjv();
  }

  virtual void CalcGrad(std::vector<Real>* grad) {
    CHECK(init_);
    std::vector<Real> dual(data_.size);
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < data_.size; ++i) {
      Real y = data_.label[i] > 0 ? 1 : -1;
      dual[i] = - y / ( 1 + exp ( y * Xw_[i] ));
    }
    SpMV::TransTimes(data_, dual, grad, nt_);
  }
};

template <typename Real>
class SquareHingeLoss : public ScalarLoss<Real> {
 public:
  using ScalarLoss<Real>::data_;
  using ScalarLoss<Real>::Xw_;
  using ScalarLoss<Real>::nt_;
  using ScalarLoss<Real>::init_;

  virtual void Evaluate(Progress* prog) {
    ScalarLoss<Real>::Evaluate(prog);
    Real objv = 0;
#pragma omp parallel for reduction(+:objv) num_threads(nt_)
    for (size_t i = 0; i < data_.size; ++i) {
      Real y = data_.label[i] > 0 ? 1 : -1;
      Real tmp = std::max(1 - y * Xw_[i], (Real)0);
      objv += tmp * tmp;
    }
    prog->objv() = objv;
  }

  virtual void CalcGrad(std::vector<Real>* grad) {
    CHECK(init_);

    std::vector<Real> dual(data_.size);
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < data_.size; ++i) {
      Real y = data_.label[i] > 0 ? 1 : -1;
      dual[i] = y * (y * Xw_[i] > 1.0);
    }
    SpMV::TransTimes(data_, dual, grad, nt_);

#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < grad->size(); ++i) {
      (*grad)[i] *= -2.0;
    }
  }
};

template <typename Real>
static ScalarLoss<Real>* CreateLoss(Config::Loss loss) {
  switch (loss) {
    case Config::LOGIT:
      return new LogitLoss<Real>();
    case Config::SQUARE_HINGE:
      return new SquareHingeLoss<Real>();
    default:
      LOG(FATAL) << "unknown type: " << loss;
  }
  return NULL;
}

}  // namespace linear
}  // namespace dmlc

#pragma once
#include <dmlc/data.h>
#include <dmlc/io.h>
#include <math.h>
#include "config.pb.h"
#include "base/spmv.h"
#include "base/binary_class_evaluation.h"
#include "linear.h"
namespace dmlc {
namespace linear {

/**
 * \brief Scalar loss
 *
 * a loss which takes as input a real value prediction and a
 * real valued label and outputs a non-negative loss value. Examples include the
 * hinge hinge loss, binary classification loss, and univariate regression loss.
 */
class ScalarLoss {
 public:
  ScalarLoss() : init_(false), nthreads_(2) { }
  virtual ~ScalarLoss() { }

  void Init(const RowBlock<unsigned>& data,
            const std::vector<Real>& w, int nt) {
    data_ = data;
    Xw_.resize(data_.size);
    SpMV::Times(data_, w, &Xw_, nthreads_);
    nthreads_ = nt;
    init_ = true;
  }

  /*! \brief evaluate the loss value */
  virtual void Evaluate(Progress* prog) {
    CHECK(init_);
    BinClassEval<Real> eval(data_.label, Xw_.data(), Xw_.size(), nthreads_);
    prog->auc()     = eval.AUC();
    prog->acc()     = eval.Accuracy(0);
    prog->num_ex()  = data_.size;
    prog->count()   = 1;
  }

  /*! \brief compute the gradients */
  virtual void CalcGrad(std::vector<Real>* grad) = 0;

  void SavePrediction(Stream* fo) {
    CHECK(init_);
    ostream os(fo);
    for (auto p : Xw_) os << p << "\n";
  }

 protected:
  bool init_;
  RowBlock<unsigned> data_;
  std::vector<Real> Xw_;  // X * w
  int nthreads_;
};

class LogitLoss : public ScalarLoss {
 public:
  virtual void Evaluate(Progress* prog) {
    ScalarLoss::Evaluate(prog);
    BinClassEval<Real> eval(data_.label, Xw_.data(), Xw_.size(), nthreads_);
    prog->objv() = eval.LogitObjv();
  }

  virtual void CalcGrad(std::vector<Real>* grad) {
    CHECK(this->init_);
    std::vector<Real> dual(data_.size);
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < data_.size; ++i) {
      Real y = data_.label[i] > 0 ? 1 : -1;
      dual[i] = - y / ( 1 + exp ( y * Xw_[i] ));
    }
    SpMV::TransTimes(data_, dual, grad, nthreads_);
  }
};

class SquareHingeLoss : public ScalarLoss {
 public:
  virtual void Evaluate(Progress* prog) {
    ScalarLoss::Evaluate(prog);
    Real objv = 0;
#pragma omp parallel for reduction(+:objv) num_threads(nthreads_)
    for (size_t i = 0; i < data_.size; ++i) {
      Real y = data_.label[i] > 0 ? 1 : -1;
      Real tmp = std::max(1 - y * Xw_[i], (Real)0);
      objv += tmp * tmp;
    }
    prog->objv() = objv;
  }

  virtual void CalcGrad(std::vector<Real>* grad) {
    CHECK(this->init_);

    std::vector<Real> dual(data_.size);
#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < data_.size; ++i) {
      Real y = data_.label[i] > 0 ? 1 : -1;
      dual[i] = y * (y * Xw_[i] > 1.0);
    }
    SpMV::TransTimes(data_, dual, grad, nthreads_);

#pragma omp parallel for num_threads(nthreads_)
    for (size_t i = 0; i < grad->size(); ++i) {
      (*grad)[i] *= -2.0;
    }
  }
};

static ScalarLoss* CreateLoss(Config::Loss loss) {
  switch (loss) {
    case Config::LOGIT:
      return new LogitLoss();
    case Config::SQUARE_HINGE:
      return new SquareHingeLoss();
    default:
      LOG(FATAL) << "unknown type: " << loss;
  }
  return NULL;
}

}  // namespace linear
}  // namespace dmlc

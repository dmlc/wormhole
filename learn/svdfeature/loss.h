/* 
 * File:   loss.h
 * Author: hexi
 *
 * Created on 2015年12月26日, 下午1:02
 */

#pragma once
#include <dmlc/data.h>
#include <dmlc/io.h>
#include <math.h>
#include "config.pb.h"
#include "progress.h"
#include "base/spmv.h"
#include "base/binary_class_evaluation.h"
#include "base/minibatch_iter.h"

namespace dmlc {
namespace svdfeature {

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
  ScalarLoss() : init_(false), bias_(0) { }
  virtual ~ScalarLoss() { }

  /**
   * \brief init
   *
   * @param data X and Y
   * @param featype feature type: 0-user,1-item,2-global
   * @param w weight
   * @param w_siz weight size
   * @param nt num of threads
   * @param dim dim
   */
  void Init(const RowBlock<unsigned>& data,
            const std::vector<float>& featype,
            const std::vector<V>& w, const std::vector<int>& w_siz, int nt, int dim) {
    Xw_.resize(data.size);
    py_.resize(data.size);
    
    std::vector<unsigned> pos;
    
    pos.resize(w_siz.size());
    unsigned p = 0;
    for (size_t i = 0; i < w_siz.size(); ++i) {
        if (w_siz[i] == 0) {
            pos[i] = (unsigned)-1;
        } else {
            pos[i] = p; p += w_siz[i];
        }
    }
    CHECK_EQ((size_t)p, w.size());
    
    trans_data_.index.resize(w.size());
    trans_data_.value.resize(w.size());
    trans_data_.offset.resize(data.size+1);
    trans_data_.label.resize(data.size);
    trans_data_.offset[0] = 0;
    std::vector<std::vector<V> > vec_sum(2);
    vec_sum[0].resize(dim,0);
    vec_sum[1].resize(dim,0);
    size_t i,j;
    int total_key_cnt = 0;
    for(i = 0; i < data.size; ++i ) {
        int key_cnt = 0;
        real_t value = 1.0f;
        py_[i] = Xw_[i] = bias_; //note: a bias should be here
        for(size_t k = 0; k < dim; k++) {
            vec_sum[0][k] = 0;
            vec_sum[1][k] = 0;
        }
        for(j = data.offset[i]; j < data.offset[i+1]; ++j) {
            key_cnt += w_siz[data.index[j]];
            value = (data.value != NULL ? data.value[j] : 1.0);
            if(static_cast<int>(featype[data.index[j]]) < 2 && w_siz[data.index[j]] == dim+1){
                V* a = BeginPtr(vec_sum[static_cast<int>(featype[data.index[j]])]);
                const V* b = BeginPtr(w)+pos[data.index[j]]+1;
                for(size_t k = 0; k < dim; k++) {
                   a[k] += b[k] * value;
                }
            }
            Xw_[i] += w[pos[data.index[j]]] * value;
            py_[i] = Xw_[i];
        }
        for(size_t k = 0; k < dim; k++) {
            py_[i] += vec_sum[0][k] * vec_sum[1][k];
        }
        
        total_key_cnt += key_cnt;
        if(trans_data_.index.size() < trans_data_.offset[i]+ key_cnt) {
            trans_data_.index.resize(trans_data_.offset[i]+ key_cnt);
            trans_data_.value.resize(trans_data_.offset[i]+ key_cnt);
        }
        size_t new_offset = trans_data_.offset[i];
        for(j = data.offset[i]; j < data.offset[i+1]; ++j) {
            value = (data.value != NULL ? data.value[j] : 1.0);
            if(w_siz[data.index[j]] == dim+1 && static_cast<int>(featype[data.index[j]]) < 2 ) {
                trans_data_.index[new_offset] = pos[data.index[j]];
                trans_data_.value[new_offset] = value;
                new_offset++;
                for(size_t k = 0; k < dim; ++k) {
                    trans_data_.index[new_offset+k] = pos[data.index[j]] + k + 1;
                    trans_data_.value[new_offset+k] = value * vec_sum[!static_cast<int>(featype[data.index[j]])][k];
                }
                new_offset += dim;         
            } else {
                trans_data_.index[new_offset] = pos[data.index[j]];
                trans_data_.value[new_offset] = value;
                new_offset++;
            }
        }
        CHECK_EQ(trans_data_.offset[i]+(size_t)key_cnt, new_offset);
        trans_data_.offset[i+1] = trans_data_.offset[i]+ key_cnt;
        trans_data_.label[i] = data.label[i];
    }
 
    data_ = trans_data_.GetBlock();
    
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
      for (auto p : py_) os << 1.0 / (1.0 + exp( - p )) << "\n";
    } else {
      for (auto p : py_) os << p << "\n";
    }
  }
  
  /*! \brief transform bias value according to loss fun */
  virtual void UseBias(float value) {
      bias_ = value;
  }

 protected:
  bool init_;
  RowBlock<unsigned> data_;
  dmlc::data::RowBlockContainer<unsigned> trans_data_;  
  std::vector<V> Xw_;  // X * w
  std::vector<V> py_;  
  int nt_;
  float bias_ ;
};

/**
 * \brief binary classification with label y = +1 / -1
 */
template <typename V>
class BinClassLoss : public ScalarLoss<V> {
 public:
  using ScalarLoss<V>::data_;
  using ScalarLoss<V>::py_;
  using ScalarLoss<V>::nt_;
  virtual void Evaluate(Progress* prog) {
    ScalarLoss<V>::Evaluate(prog);
    BinClassEval<V> eval(data_.label, py_.data(), py_.size(), nt_);
    prog->auc()     = eval.AUC();
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
  using ScalarLoss<V>::py_;
  using ScalarLoss<V>::nt_;
  using ScalarLoss<V>::init_;
  using ScalarLoss<V>::bias_;

  virtual void Evaluate(Progress* prog) {
    BinClassLoss<V>::Evaluate(prog);
    BinClassEval<V> eval(data_.label, Xw_.data(), Xw_.size(), nt_);
    prog->objv_w() = eval.LogitObjv();
    BinClassEval<V> eval2(data_.label, py_.data(), py_.size(), nt_);
    prog->objv() = eval2.LogitObjv();
  }

  virtual void CalcGrad(std::vector<V>* grad) {
    CHECK(init_);
    std::vector<V> dual(data_.size);
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < data_.size; ++i) {
      V y = data_.label[i] > 0 ? 1 : -1;
      dual[i] = - y / ( 1 + exp ( y * py_[i] ));
    }
    SpMV::TransTimes(data_, dual, grad, nt_);
  }
  
  /*! \brief transform bias value according to loss fun */
  virtual void UseBias(float value) {
     CHECK(value > 0.0f && value < 1.0f) << "sigmoid range constrain";
     bias_ =  - logf( 1.0f / value - 1.0f );
  }
  
};

/**
 * \brief square hinge loss: \f$ \max\left(0, (1-yp)^2\right) \f$
 */
template <typename V>
class SquareHingeLoss : public BinClassLoss<V> {
 public:
  using ScalarLoss<V>::data_;
  using ScalarLoss<V>::py_;
  using ScalarLoss<V>::Xw_;
  using ScalarLoss<V>::nt_;
  using ScalarLoss<V>::init_;

  virtual void Evaluate(Progress* prog) {
    BinClassLoss<V>::Evaluate(prog);
    V objv = 0;
    V objv_w = 0;
#pragma omp parallel for reduction(+:objv) reduction(+:objv_w) num_threads(nt_)
    for (size_t i = 0; i < data_.size; ++i) {
      V y = data_.label[i] > 0 ? 1 : -1;
      V tmp = std::max(1 - y * py_[i], (V)0);
      objv += tmp * tmp;
      tmp = std::max(1 - y * Xw_[i], (V)0);
      objv_w += tmp * tmp;
    }
    prog->objv() = objv;
    prog->objv_w() = objv_w;
  }

  virtual void CalcGrad(std::vector<V>* grad) {
    CHECK(init_);

    std::vector<V> dual(data_.size);
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < data_.size; ++i) {
      V y = data_.label[i] > 0 ? 1 : -1;
      dual[i] = y * (y * py_[i] > 1.0);
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
  using ScalarLoss<V>::data_;
  using ScalarLoss<V>::py_;
  using ScalarLoss<V>::Xw_;
  using ScalarLoss<V>::nt_;
  using ScalarLoss<V>::init_;
    
  virtual void Evaluate(Progress* prog) {
    ScalarLoss<V>::Evaluate(prog);
    V objv = 0;
    V objv_w = 0;
#pragma omp parallel for reduction(+:objv) reduction(+:objv_w) num_threads(nt_)
    for (size_t i = 0; i < data_.size; ++i) {
      V y = data_.label[i];
      V tmp = py_[i]-y;
      objv += 0.5 * tmp * tmp;
      tmp = Xw_[i]-y;
      objv_w += 0.5 * tmp * tmp;
    }
    prog->objv() = objv;
    prog->objv_w() = objv_w;
  }
  
 virtual void CalcGrad(std::vector<V>* grad) {
    CHECK(init_);
    std::vector<V> dual(data_.size);
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < data_.size; ++i) {
        dual[i] = py_[i] - data_.label[i];
    }
    SpMV::TransTimes(data_, dual, grad, nt_);
  }
 
   /**
   * \brief save prediction
   * \param prob_out output probability
   */
  virtual void Predict(Stream* fo, bool prob_out) {
    CHECK(init_); CHECK_NOTNULL(fo);
    ostream os(fo);
    for (auto p : py_) os << p << "\n";
  }
  
};

/**
 * \brief loss factory
 */
template <typename V>
static ScalarLoss<V>* CreateLoss(Config::Loss loss) {
  switch (loss) {
    case Config::SQUARE:
      return new SquareLoss<V>();
    case Config::LOGIT:
      return new LogitLoss<V>();
    case Config::SQUARE_HINGE:
      return new SquareHingeLoss<V>();
    default:
      LOG(FATAL) << "unknown type: " << loss;
  }
  return NULL;
}

}  // namespace svdfeature
}  // namespace dmlc


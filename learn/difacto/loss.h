#pragma once
#include "base/spmm.h"
#include "base/binary_class_evaluation.h"
#include "config.pb.h"
#include "dmlc/data.h"
#include "dmlc/io.h"
namespace dmlc {
namespace difacto {

/**
 * \brief the loss function
 */
template <typename T>
class Loss {
 public:
  /**
   * create and init the loss function
   *
   * @param data X and Y
   * @param model w and V
   * @param model_siz 1 + length V[i]
   * @param conf difacto conf
   */
  Loss(const RowBlock<unsigned>& data,
       const std::vector<T>& model,
       const std::vector<int>& model_siz,
       const Config& conf) {
    nt_ = conf.num_threads();

    // init w
    w.Load(0, data, model, model_siz);

    // init V
    if (conf.embedding_size() == 0) return;
    const auto& cf = conf.embedding(0);
    if (cf.dim() == 0) return;
    V.Load(cf.dim(), data, model, model_siz);
    V.dropout            = cf.dropout();
    V.grad_clipping      = cf.grad_clipping();
    V.grad_normalization = cf.grad_normalization();
  }

  ~Loss() { }

  /**
   * \brief evaluate the progress
   * predict_y
   *  py = X * w + .5 * sum((X*V).^2 - (X.*X)*(V.*V), 2);
   *
   * sum(A, 2) : sum the rows of A
   * .* : elemenetal-wise times
   */
  void Evaluate(Progress* prog) {

    // py = X * w
    py_.resize(w.X.size);
    SpMV::Times(w.X, w.weight, &py_, nt_);

    BinClassEval<T> eval(w.X.label, py_.data(), py_.size(), nt_);
    prog->objv_w() = eval.LogitObjv();

    // py += .5 * sum((X*V).^2 - (X.*X)*(V.*V), 2);
    if (!V.weight.empty()) {
      // tmp = (X.*X)*(V.*V)
      std::vector<T> vv = V.weight;
      for (auto& v : vv) v *= v;
      CHECK_EQ(vv.size(), V.pos.size() * V.dim);
      std::vector<T> xxvv(V.X.size * V.dim);
      SpMM::Times(V.XX, vv, &xxvv, nt_);

      // V.XV = X*V
      V.XV.resize(xxvv.size());
      SpMM::Times(V.X, V.weight, &V.XV, nt_);

      // py += .5 * sum((V.XV).^2 - xxvv)
#pragma omp parallel for num_threads(nt_)
      for (size_t i = 0; i < py_.size(); ++i) {
        T* t = V.XV.data() + i * V.dim;
        T* tt = xxvv.data() + i * V.dim;
        T s = 0;
        for (int j = 0; j < V.dim; ++j) s += t[j] * t[j] - tt[j];
        py_[i] += .5 * s;
      }
    }

    // auc, acc, logloss, copc
    prog->objv()   = eval.LogitObjv();
    prog->auc()    = eval.AUC();
    // prog->copc()   = eval.Copc();
    prog->new_ex() = w.X.size;
    prog->count()  = 1;
  }

  /*!
   * \brief compute the gradients
   * p = - y ./ (1 + exp (y .* py));
   * grad_w = X' * p;
   * grad_u = X' * diag(p) * X * V  - diag((X.*X)'*p) * V
   */
  void CalcGrad(std::vector<T>* grad) {
    // p = ... (reuse py_)
    CHECK_EQ(py_.size(), w.X.size) << "call *evaluate* first";
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < py_.size(); ++i) {
      T y = w.X.label[i] > 0 ? 1 : -1;
      py_[i] = - y / ( 1 + exp ( y * py_[i] ));
    }

    // grad_w = ...
    SpMV::TransTimes(w.X, py_, &w.weight, nt_);
    w.Save(grad);

    // grad_u = ...
    if (!V.weight.empty()) {
      int dim = V.dim;

      // xxp = (X.*X)'*p
      size_t m = V.pos.size();
      std::vector<T> xxp(m);
      SpMM::TransTimes(V.XX, py_, &xxp, nt_);

      // V = - diag(xxp) * V
      CHECK_EQ(V.weight.size(), dim * m);
#pragma omp parallel for num_threads(nt_)
      for (size_t i = 0; i < m; ++i) {
        T* v = V.weight.data() + i * dim;
        for (int j = 0; j < dim; ++j) v[j] *= - xxp[i];
      }

      // V.XV = diag(p) * X * V
      size_t n = py_.size();
      CHECK_EQ(V.XV.size(), n * dim);
#pragma omp parallel for num_threads(nt_)
      for (size_t i = 0; i < n; ++i) {
        T* y = V.XV.data() + i * dim;
        for (int j = 0; j < dim; ++j) y[j] *= py_[i];
      }

      // V += X' * V.XV
      SpMM::TransTimes(V.X, V.XV, (T)1, V.weight, &V.weight, nt_);

      // some preprocessing
      if (V.grad_clipping > 0) {
        T gc = V.grad_clipping;
        for (T& g : V.weight) g = g > gc ? gc : ( g < -gc ? -gc : g);
      }

      if (V.dropout > 0) {
        for (T& g : V.weight) {
          if ((T)rand() / RAND_MAX > 1 - V.dropout) g = 0;
        }
      }
      if (V.grad_normalization) Normalize(V.weight);
    }
    V.Save(grad);
  }

  void Normalize(std::vector<T>& grad) {
    T norm = 0;
    for (T g : grad) norm += g * g;
    if (norm < 1e-10) return;
    norm = sqrt(norm);
    for (T& g : grad) g = g / norm;
  }

  virtual void Predict(Stream* fo, bool prob_out) {
    ostream os(fo);
    if (prob_out) {
      for (auto p : py_) os << 1.0 / (1.0 + exp( - p )) << "\n";
    } else {
      for (auto p : py_) os << p << "\n";
    }
  }

 private:
  /// \brief store data and model w (dim==0) and V (dim >= 1)
  struct Data {
    /// \brief get data and model
    void Load(int d, const RowBlock<unsigned>& data,
              const std::vector<T>& model,
              const std::vector<int>& model_siz) {
      // init pos and w
      std::vector<unsigned> col_map;
      dim = d;
      if (dim == 0) {  // w
        pos.resize(model_siz.size());
        weight.resize(model_siz.size());
        unsigned p = 0;
        for (size_t i = 0; i < model_siz.size(); ++i) {
          if (model_siz[i] == 0) {
            pos[i] = (unsigned)-1;
          } else {
            pos[i] = p; weight[i] = model[p]; p += model_siz[i];
          }
        }
        CHECK_EQ((size_t)p, model.size());
      } else {  // V
        col_map.resize(model_siz.size());
        unsigned k = 0, p = 0;
        for (size_t i = 0; i < model_siz.size(); ++i) {
          if (model_siz[i] == dim + 1) {
            pos.push_back(p+1);  // skip the first dim
            col_map[i] = ++ k;
          }
          p += model_siz[i];
        }
        CHECK_EQ((size_t)p, model.size());
        weight.resize(pos.size() * dim);
        for (size_t i = 0; i < pos.size(); ++i) {
          memcpy(weight.data()+i*dim, model.data()+pos[i], dim*sizeof(V));
        }
      }
      if (weight.empty()) return;

      // init X
      if (dim == 0) {  // w
        X = data;
      } else {  // V
        // pick the columns with model_siz = dim + 1
        os.push_back(0);
        for (size_t i = 0; i < data.size; ++i) {
          for (size_t j = data.offset[i]; j < data.offset[i+1]; ++j) {
            unsigned d = data.index[j];
            unsigned k = col_map[d];
            if (k > 0) {
              idx.push_back(k-1);
              if (data.value) val_.push_back(data.value[j]);
            }
          }
          os.push_back(idx.size());
        }
        X.size = data.size;
        X.offset = BeginPtr(os);
        X.value = BeginPtr(val_);
        X.index = BeginPtr(idx);
      }

      // init XX
      XX = X;
      if (X.value) {
        val2_.resize(X.offset[X.size]);
        for (size_t i = 0; i < val2_.size(); ++i) {
          val2_[i] = X.value[i] * X.value[i];
        }
        XX.value = BeginPtr(val2_);
      }
    }

    /// \brief set the gradient
    void Save(std::vector<T>* grad) const {
      if (weight.empty()) return;
      int d = dim == 0 ? 1 : dim;
      CHECK_EQ(weight.size(), pos.size()*d);
      for (size_t i = 0; i < pos.size(); ++i) {
        if (pos[i] == (unsigned)-1) continue;
        memcpy(grad->data()+pos[i], weight.data()+i*d, d*sizeof(V));
      }
    }

    int dim;
    RowBlock<unsigned> X, XX;  // XX = X.*X
    std::vector<T> weight;
    std::vector<unsigned> pos;

    std::vector<T> XV;
    T dropout = 0;
    T grad_clipping = 0;
    T grad_normalization = 0;
   private:
    std::vector<T> val_, val2_;
    std::vector<size_t> os;
    std::vector<unsigned> idx;
  };
  Data w, V;
  // std::vector<Data> data_;

  std::vector<T> py_;
  int nt_;  // number of threads
};

}  // namespace difacto
}  // namespace dmlc

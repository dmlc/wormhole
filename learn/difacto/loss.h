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
template <typename V>
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
       const std::vector<V>& model,
       const std::vector<int>& model_siz,
       const Config& conf) {
    nt_ = conf.num_threads();

    Config cf; cf.add_embedding()->set_dim(0);
    for (int i = 0; i < conf.embedding_size(); ++i) {
      if (conf.embedding(i).dim() > 0) {
        cf.add_embedding()->CopyFrom(conf.embedding(i));
      }
    }

    int n = cf.embedding_size();
    data_.resize(n);

    for (int i = 0; i < n; ++i) {
      const auto& eb = cf.embedding(i);
      data_[i].Load(eb.dim(), data, model, model_siz);
      if (i > 0) CHECK_GT(data_[i].dim, data_[i-1].dim);

      data_[i].dropout = eb.dropout();
      data_[i].grad_clipping = eb.grad_clipping();
      data_[i].grad_normalization = eb.grad_normalization();
    }
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
    auto& d = data_[0];
    py_.resize(d.X.size);
    SpMV::Times(d.X, d.w, &py_, nt_);

    BinClassEval<V> eval(d.X.label, py_.data(), py_.size(), nt_);
    prog->objv_w() = eval.LogitObjv();

    // py += .5 * sum((X*V).^2 - (X.*X)*(V.*V), 2);
    for (size_t k = 1; k < data_.size(); ++k) {
      auto& d = data_[k];
      if (d.w.empty()) continue;

      // tmp = (X.*X)*(V.*V)
      std::vector<V> vv = d.w;
      for (auto& v : vv) v *= v;
      CHECK_EQ(vv.size(), d.pos.size() * d.dim);
      std::vector<V> xxvv(d.X.size * d.dim);
      SpMM::Times(d.XX, vv, &xxvv, nt_);

      // d.XV = X*V
      d.XV.resize(xxvv.size());
      SpMM::Times(d.X, d.w, &d.XV, nt_);

      // py += .5 * sum((d.XV).^2 - xxvv)
#pragma omp parallel for num_threads(nt_)
      for (size_t i = 0; i < py_.size(); ++i) {
        V* t = d.XV.data() + i * d.dim;
        V* tt = xxvv.data() + i * d.dim;
        V s = 0;
        for (int j = 0; j < d.dim; ++j) s += t[j] * t[j] - tt[j];
        py_[i] += .5 * s;
      }
    }

    // auc, acc, logloss, copc
    prog->objv()   = eval.LogitObjv();
    prog->auc()    = eval.AUC();
    // prog->copc()   = eval.Copc();
    prog->new_ex() = d.X.size;
    prog->count()  = 1;
  }

  /*!
   * \brief compute the gradients
   * p = - y ./ (1 + exp (y .* py));
   * grad_w = X' * p;
   * grad_u = X' * diag(p) * X * V  - diag((X.*X)'*p) * V
   */
  void CalcGrad(std::vector<V>* grad) {
    // p = ... (reuse py_)
    auto& d = data_[0];
    CHECK_EQ(py_.size(), d.X.size) << "call *evaluate* first";
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < py_.size(); ++i) {
      V y = d.X.label[i] > 0 ? 1 : -1;
      py_[i] = - y / ( 1 + exp ( y * py_[i] ));
    }

    // grad_w = ...
    SpMV::TransTimes(d.X, py_, &d.w, nt_);
    // Normalize(d.w);

    // grad_u = ...
    for (size_t k = 1; k < data_.size(); ++k) {
      auto& d = data_[k];
      if (d.w.empty()) continue;
      int dim = d.dim;

      // xxp = (X.*X)'*p
      size_t m = d.pos.size();
      std::vector<V> xxp(m);
      SpMM::TransTimes(d.XX, py_, &xxp, nt_);

      // V = - diag(xxp) * V
      CHECK_EQ(d.w.size(), dim * m);
#pragma omp parallel for num_threads(nt_)
      for (size_t i = 0; i < m; ++i) {
        V* v = d.w.data() + i * dim;
        for (int j = 0; j < dim; ++j) v[j] *= - xxp[i];
      }

      // d.XV = diag(p) * X * V
      size_t n = py_.size();
      CHECK_EQ(d.XV.size(), n * dim);
#pragma omp parallel for num_threads(nt_)
      for (size_t i = 0; i < n; ++i) {
        V* y = d.XV.data() + i * dim;
        for (int j = 0; j < dim; ++j) y[j] *= py_[i];
      }

      // V += X' * d.XV
      SpMM::TransTimes(d.X, d.XV, (V)1, d.w, &d.w, nt_);

      // some preprocessing
      if (d.grad_clipping > 0) {
        V gc = d.grad_clipping;
        for (V& g : d.w) g = g > gc ? gc : ( g < -gc ? -gc : g);
      }

      if (d.dropout > 0) {
        for (V& g : d.w) {
          if ((V)rand() / RAND_MAX > 1 - d.dropout) g = 0;
        }
      }
      if (d.grad_normalization) Normalize(d.w);
    }

    for (const auto& d : data_) d.Save(grad);
  }

  void Normalize(std::vector<V>& grad) {
    V norm = 0;
    for (V g : grad) norm += g * g;
    if (norm < 1e-10) return;
    norm = sqrt(norm);
    for (V& g : grad) g = g / norm;
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
              const std::vector<V>& model,
              const std::vector<int>& model_siz) {
      // init pos and w
      std::vector<unsigned> col_map;
      dim = d;
      if (dim == 0) {  // w
        pos.resize(model_siz.size());
        w.resize(model_siz.size());
        unsigned p = 0;
        for (size_t i = 0; i < model_siz.size(); ++i) {
          if (model_siz[i] == 0) {
            pos[i] = (unsigned)-1;
          } else {
            pos[i] = p; w[i] = model[p]; p += model_siz[i];
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
        w.resize(pos.size() * dim);
        for (size_t i = 0; i < pos.size(); ++i) {
          memcpy(w.data()+i*dim, model.data()+pos[i], dim*sizeof(V));
        }
      }
      if (w.empty()) return;

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
              if (data.value) val.push_back(data.value[j]);
            }
          }
          os.push_back(idx.size());
        }
        X.size = data.size;
        X.offset = BeginPtr(os);
        X.value = BeginPtr(val);
        X.index = BeginPtr(idx);
      }

      // init XX
      XX = X;
      if (X.value) {
        v2.resize(X.offset[X.size]);
        for (size_t i = 0; i < v2.size(); ++i) v2[i] = X.value[i] * X.value[i];
        XX.value = BeginPtr(v2);
      }
    }

    /// \brief set the gradient
    void Save(std::vector<V>* grad) const {
      if (w.empty()) return;
      int d = dim == 0 ? 1 : dim;
      CHECK_EQ(w.size(), pos.size()*d);
      for (size_t i = 0; i < pos.size(); ++i) {
        if (pos[i] == (unsigned)-1) continue;
        memcpy(grad->data()+pos[i], w.data()+i*d, d*sizeof(V));
      }
    }

    int dim;
    RowBlock<unsigned> X, XX;  // XX = X.*X
    std::vector<V> w;
    std::vector<unsigned> pos;

    std::vector<V> XV;
    V dropout = 0;
    V grad_clipping = 0;
    V grad_normalization = 0;
   private:
    std::vector<V> val, v2;
    std::vector<size_t> os;
    std::vector<unsigned> idx;
  };
  // Data w, V;
  std::vector<Data> data_;

  std::vector<V> py_;
  int nt_;  // number of threads
};

}  // namespace difacto
}  // namespace dmlc

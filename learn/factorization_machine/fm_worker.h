#pragma once
#include "fm.h"
#include "base/spmm.h"
#include "base/localizer.h"
#include "base/evaluation.h"

namespace dmlc {
namespace fm {

////////////////////////////////////////////////////////////
//  objective
////////////////////////////////////////////////////////////

class Objective {
 public:
  Objective(const RowBlock<unsigned>& data,
            const std::vector<Real>& model,
            const std::vector<int>& model_siz,
            const std::vector<int>& dims,
            int num_threads = 2) {
    nt_ = num_threads;
    CHECK_EQ(dims[0], 1);
    data_.resize(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      data_[i].Load(data, model, model_siz);
    }
  }

  ~Objective() { }

  /**
   * \brief evaluate the progress
   * predict_y
   *  py = x * w + .5 * sum((x*u).^2 + (x.*x)*(u.*u), 2);
   */
  void Evaluate(Progress* prog) {
    // py = X * w
    auto& d = data_[0];
    py.resize(d.X.size);
    SpMV::Times(d.X, d.w, &py, nt_);

    for (size_t k = 1; k < data_.size(); ++k) {
      // py += .5 * sum((x*u).^2 + (x.*x)*(u.*u), 2);
      auto& d = data[k];
      if (d.w.empty()) continue;

      // (x.*x)*(u.*u)
      std::vector<Real> uu = d.x;
#pragma omp for num_threads(nt_)
      for (auto& u : uu) u *= u;

      std::vector<Real> tmp(d.X.size * d.dim);
      SpMM::Times(d.XX, uu, &tmp, nt_);

      // x*u
      d.xw.resize(tmp.size();
      SpMM::Times(d.X, d.w, &d.xw, nt_);

      // (x*u).^2
#pragma omp for num_threads(nt_)
      for (auto& t : tmp) t *= t;

      // py += .5 * sum(..., 2)
#pragma omp for num_threads(nt_)
      for (size_t i = 0; i < py.size(); ++i) {
        Real* tt = tmp.data() + i * d.dim;
        Real* t = d.xw.data() + i * d.dim;
        Real s = 0;
        for (size_t j = 0; j < d.dim; ++j) s += t[j] * t[j] + tt[j];
        py[i] += .5 * s;
      }
    }

    // auc, acc, logloss,
    BinClassEval<Real> eval(d.X.label, py_.data(), py_.size(), nt_);
    prog->objv() = eval.LogitObjv();
    prog->auc() = eval.AUC();
    prog->acc() = eval.Accuracy();
    prog->logloss() = eval.LogLoss();
  }

  /*!
   * \brief compute the gradients
   * p = - y ./ (1 + exp (y .* py));
   * grad_w = x' * p;
   * grad_u = x' * bsxfun(@times, p, x*u) - bsxfun(@times, (x.*x)'*p, u)
   */
  void CalcGrad(std::vector<Real>* grad) {
    CHECK(py_.size()) << "call *evaluate* first";
    auto& d = data_[0];
#pragma omp for num_threads(nt_)
    for (size_ i = 0; i < py_.size(); ++i) {
      Real y = d.label[i] > 0 ? 1 : -1;
      py_[i] = - y / ( 1 + exp ( y * py_[i] ));
    }

    // grad_w
    SpMV::TransTimes(d.X, py_, &d.w, nt_);

    // grad_u
    for (size_t k = 1; k < data_.size(); ++k) {
      auto& d = data[k];
      if (d.w.empty()) continue;
      int dim = d.dim;

      // (x.*x)'*p
      size_t m = d.pos.size();
      std::vector<Real> xxp(m);
      SpMM::TransTimes(d.XX, py_, &xxp, nt_);

      // bsxfun(@times, (x.*x)'*p, u)
#pragma omp for num_threads(nt_)
      for (size_t i = 0; i < m; ++i) {
        Real* u = d.w.data() + i * dim;
        for (int j = 0; j < dim; ++j) u[j] *= -xxp[i];
      }

      // bsxfun(@times, p, x*u)
      size_t n = py_.size();
#pragma omp for num_threads(nt_)
      for (size_t i = 0; i < n; ++i) {
        Real* y = d.xw.data() + i * dim;
        for (int j = 0; j < dim; ++j) y[i] *= py_[i];
      }

      // += x' * bsxfun(@times, p, x*u)
      SpMM::TransTimes(d.X, d.xw, 1, d.w, &d.w, nt_);
    }

    for (const auto& d : data_) d.Save(grad);
  }

 private:
  // store w (dim==1) and u (dim > 1)
  struct Data {
    Data() { }
    ~Data() {
      if (dim != 1) {
        delete [] X.offset;
        delete [] X.index;
        delete [] X.value;
      }
      delete [] XX.value;
    }
    void Load(int dimension,
              const RowBlock<unsigned>& data,
              const std::vector<Real>& model,
              const std::vector<int>& model_siz) {
      // pos and w
      std::vector<bool> hit_cols;
      dim = dimension;
      if (dim == 1) {
        pos.reserve(model_siz.size());
        w.reserve(model_siz.size());
        unsigned p = 0;
        for (int i : model_siz) {
          CHECK_GE(i, 1);
          pos.push_back(p);
          w.push_back(model[p]);
          p += i;
        }
        CHECK_EQ((size_t)p, model.size());
      } else {
        hit_cols.resize(model_siz.size());
        unsigned p = 0;
        for (size_t i = 0; i < model_siz; ++i) {
          if (model_siz[i] == dim) {
            pos.push_back(p);
            hit_cols[i] = true;
          }
          p += model_siz[i];
        }
        w.resize(pos.size() * dim);
        for (size_t i = 0; i < pos.size(); ++i) {
          memcpy(w.data() + i * dim, model.data() + pos[i] * dim, dim * sizeof(Real));
        }
      }

      if (w.empty()) return;

      // X
      if (dim == 1) {
        X = data;
      } else {
        // slice data
        size_t n = data.offset[data.size];
        bool val = data.value;
        X.size = data.size;
        X.offset = new size_t[X.size+1];
        X.index = new unsigned[n];
        X.value = val ? new Real[n] : NULL;

        for (size_t i = 0; i < X.size; ++i) {
          size_t os = X.offset[i];
          for (size_t j = data.offset[i]; j < data.offset[i+1]; ++j) {
            if (hit_cols[data.index[j]]) {
              X.index[os] = data.index[j];
              if (val) X.value[os] = data.value[j];
            }
          }
          X.offset[i+1] = os;
        }
      }

      // XX
      XX = X;
      if (X.value) {
        size_t n = X.offset[X.size];
        XX.value = new Real[n];
        for (size_t i = 0; i < n; ++i) XX.value[i] = X.value[i] * X.value[i];
      }
    }

    void Save(std::vector<Real>* grad) const {
      if (w.empty()) return;
      for (size_t i = 0; i < pos.size(); ++i) {
        memcpy(w.data() + i*dim, grad + pos[i], dim * sizeof(Real));
      }
    }

    int dim;
    RowBlock<unsigned> X;
    RowBlock<unsigned> XX;
    std::vector<Real> w;
    std::vector<unsigned> pos;

    std::vector<Real> xw;

    // RowBlockContainer<unsigned> dat;
  };
  std::vector<Data> data_;

  std::vector<Real> py_;
  int nt_;  // number of threads
};

////////////////////////////////////////////////////////////
//  sgd solver
////////////////////////////////////////////////////////////

class FMWorker : public solver::AsyncSGDWorker {
 public:
  FMWorker(const Config& conf) : conf_(conf) { }
  virtual ~FMWorker() { }

 protected:

  virtual void ProcessMinibatch(const Minibatch& mb, bool train) {

    auto data = new dmlc::data::RowBlockContainer<unsigned>();
    auto feaid = std::make_shared<std::vector<FeaID>>();
    auto feacnt = std::make_shared<std::vector<Real>>();

    Localizer<FeaID> lc;
    lc.Localize(mb, data, feaid.get(), feacnt.get());

    ps::SyncOpts pull_w_opts;

    if (train) {
      // push the feature count to the servers
      ps::SyncOpts push_cnt_opts;
      SetFilters(true, &push_cnt_opts);
      int t = server_.ZPush(feaid, feacnt, push_cnt_opts);
      pull_w_opts.deps.push_back(t);
    }

    // pull the weight from the servers
    auto val = new std::vector<Real>();
    auto val_siz = new std::vector<int>();

    // this callback will be called when the weight has been actually pulled back
    pull_w_opts.callback = [this, data, feaid, val, val_siz, train]() {
      // eval the objective, and report progress to the scheduler
      Objective obj(data->GetBlock(), *val, *val_siz);
      Progress prog;
      obj.Evaluate(&prog);
      Report(&prog);

      // monitor_.Update(local->label.size(), loss);

      if (train) {
        // calculate and push the gradients
        obj.CalcGrad(val, val_siz);

        ps::SyncOpts push_grad_opts;
        // filters to reduce network traffic
        SetFilters(true, &push_grad_opts);
        // this callback will be called when the gradients have been actually pushed
        push_grad_opts.callback = [this]() { FinishMinibatch(); };
        server_.ZVPush(feaid,
                       std::shared_ptr<std::vector<Real>>(val),
                       std::shared_ptr<std::vector<int>>(val_siz),
                       push_grad_opts);
      } else {
        FinishMinibatch();
        delete val;
        delete val_siz;
      }
      delete data;
    };

    // filters to reduce network traffic
    SetFilters(false, &pull_w_opts);
    server_.ZVPull(feaid, val, val_siz, pull_w_opts);
  }

 private:
  void SetFilters(bool push, ps::SyncOpts* opts) {
    if (conf_.fixed_bytes() > 0) {
      opts->AddFilter(ps::Filter::FIXING_FLOAT)->set_num_bytes(conf_.fixed_bytes());
    }
    if (conf_.key_cache()) {
      opts->AddFilter(ps::Filter::KEY_CACHING)->set_clear_cache(push);
    }
    if (conf_.msg_compression()) {
      opts->AddFilter(ps::Filter::COMPRESSING);
    }
  }
  Config conf_;
  ps::KVWorker<Real> server_;
};

}  // namespace fm
}  // namespace dmlc

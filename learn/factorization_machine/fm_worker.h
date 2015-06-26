#pragma once
#include "fm.h"
#include "base/spmm.h"
#include "base/localizer.h"
#include "base/binary_class_evaluation.h"
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
    data_.resize(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
      if (i == 0) {
        CHECK_EQ(dims[0], 1);
      } else {
        CHECK_GT(dims[i], dims[i-1]);
      }
      data_[i].Load(dims[i], data, model, model_siz);
    }
  }

  ~Objective() { }

  /**
   * \brief evaluate the progress
   * predict_y
   *  py = x * w + .5 * sum((x*u).^2 - (x.*x)*(u.*u), 2);
   */
  void Evaluate(Progress* prog) {
    // py = X * w
    auto& d = data_[0];

    py_.resize(d.X.size);
    SpMV::Times(d.X, d.w, &py_, nt_);

    // py += .5 * sum((x*u).^2 + (x.*x)*(u.*u), 2);
    for (size_t k = 1; k < data_.size(); ++k) {
      auto& d = data_[k];
      if (d.w.empty()) continue;

      // (x.*x)*(u.*u)
      std::vector<Real> uu = d.w;
      for (auto& u : uu) u *= u;

      std::vector<Real> tmp(d.X.size * d.dim);
      CHECK_EQ(uu.size(), d.pos.size()*d.dim);
      SpMM::Times(d.XX, uu, &tmp, nt_);

      // x*u
      d.xw.resize(tmp.size());
      SpMM::Times(d.X, d.w, &d.xw, nt_);

      // py += .5 * sum(..., 2)
#pragma omp parallel for num_threads(nt_)
      for (size_t i = 0; i < py_.size(); ++i) {
        Real* tt = tmp.data() + i * d.dim;
        Real* t = d.xw.data() + i * d.dim;
        Real s = 0;
        for (int j = 0; j < d.dim; ++j) s += t[j] * t[j] - tt[j];
        py_[i] += .5 * s;
      }
    }

    // auc, acc, logloss,
    BinClassEval<Real> eval(d.X.label, py_.data(), py_.size(), nt_);
    prog->objv()    = eval.LogitObjv();
    prog->auc()     = eval.AUC();
    prog->acc()     = eval.Accuracy(0);
    prog->num_ex()  = d.X.size;
    prog->count()   = 1;
  }

  /*!
   * \brief compute the gradients
   * p = - y ./ (1 + exp (y .* py));
   * grad_w = x' * p;
   * grad_u = x' * bsxfun(@times, p, x*u) - bsxfun(@times, (x.*x)'*p, u)
   */
  void CalcGrad(std::vector<Real>* grad) {
    // p =
    auto& d = data_[0];
    CHECK_EQ(py_.size(), d.X.size) << "call *evaluate* first";
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < py_.size(); ++i) {
      Real y = d.X.label[i] > 0 ? 1 : -1;
      py_[i] = - y / ( 1 + exp ( y * py_[i] ));
    }

    // grad_w =
    SpMV::TransTimes(d.X, py_, &d.w, nt_);

    // grad_u
    for (size_t k = 1; k < data_.size(); ++k) {
      auto& d = data_[k];
      if (d.w.empty()) continue;
      int dim = d.dim;

      // (x.*x)'*p
      size_t m = d.pos.size();
      std::vector<Real> xxp(m);
      SpMM::TransTimes(d.XX, py_, &xxp, nt_);

      // bsxfun(@times, (x.*x)'*p, u)
      CHECK_EQ(d.w.size(), dim * m);
#pragma omp parallel for num_threads(nt_)
      for (size_t i = 0; i < m; ++i) {
        Real* u = d.w.data() + i * dim;
        for (int j = 0; j < dim; ++j) u[j] *= - xxp[i];
      }

      // bsxfun(@times, p, x*u)
      size_t n = py_.size();
      CHECK_EQ(d.xw.size(), n * dim);
#pragma omp parallel for num_threads(nt_)
      for (size_t i = 0; i < n; ++i) {
        Real* y = d.xw.data() + i * dim;
        for (int j = 0; j < dim; ++j) y[j] *= py_[i];
      }

      // += x' * bsxfun(@times, p, x*u)
      SpMM::TransTimes(d.X, d.xw, (Real)1, d.w, &d.w, nt_);
    }

    for (const auto& d : data_) d.Save(grad);
  }

 private:
  // store w (dim==1) and u (dim > 1)
  struct Data {
    Data() { }
    ~Data() { }
    void Load(int d, const RowBlock<unsigned>& data,
              const std::vector<Real>& model,
              const std::vector<int>& model_siz) {
      // pos and w
      dim = d;
      std::vector<unsigned> col_map;
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
          memcpy(w.data() + i * dim, model.data() + pos[i], dim * sizeof(Real));
        }
      }

      if (w.empty()) return;

      // X
      if (dim == 1) {
        X = data;
      } else {
        // slice data
        os.push_back(0);
        for (size_t i = 0; i < data.size; ++i) {
          for (size_t j = data.offset[i]; j < data.offset[i+1]; ++j) {
            unsigned d = data.index[j];
            CHECK_LT(d, col_map.size());
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

      // XX
      XX = X;
      if (X.value) {
        val2.resize(X.offset[X.size]);
        for (size_t i = 0; i < val2.size(); ++i) val2[i] = X.value[i] * X.value[i];
        XX.value = BeginPtr(val2);
      }
    }

    void Save(std::vector<Real>* grad) const {
      if (w.empty()) return;
      CHECK_EQ(w.size(), pos.size()*dim);
      CHECK_GE(grad->size(), pos.back() + dim);
      for (size_t i = 0; i < pos.size(); ++i) {
        memcpy(grad->data() + pos[i], w.data() + i*dim, dim * sizeof(Real));
      }
    }

    int dim;
    RowBlock<unsigned> X;
    RowBlock<unsigned> XX;
    std::vector<Real> w;
    std::vector<unsigned> pos;

    std::vector<Real> xw;

   private:
    std::vector<Real> val, val2;
    std::vector<size_t> os;
    std::vector<unsigned> idx;
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
  FMWorker(const Config& conf) : conf_(conf) {
    minibatch_size_ = conf.minibatch();
    max_delay_      = conf.max_delay();
    nt_             = conf.num_threads();
    if (conf.use_worker_local_data()) {
      train_data_        = conf.train_data();
      val_data_          = conf.val_data();
      worker_local_data_ = true;
    }
    dims_.push_back(1);
    for (int i = 0; i < conf.embedding_size(); ++i) {
      dims_.push_back(conf.embedding(i).dim());
      CHECK_GT(dims_[i], dims_[i-1]);
    }
  }
  virtual ~FMWorker() { }

 protected:

  virtual void ProcessMinibatch(const Minibatch& mb, int data_pass, bool train) {

    auto data = new dmlc::data::RowBlockContainer<unsigned>();
    auto feaid = std::make_shared<std::vector<FeaID>>();
    auto feacnt = std::make_shared<std::vector<Real>>();

    double start = GetTime();
    Localizer<FeaID> lc;
    lc.Localize(mb, data, feaid.get(), feacnt.get());
    workload_time_ += GetTime() - start;

    ps::SyncOpts pull_w_opt;
    if (train && data_pass == 0) {
      // push the feature count to the servers
      ps::SyncOpts cnt_opt;
      SetFilters(0, &cnt_opt);
      cnt_opt.cmd = kPushFeaCnt;
      int t = server_.ZPush(feaid, feacnt, cnt_opt);
      pull_w_opt.deps.push_back(t);
      // LL << DebugStr(*feacnt);
    }

    // pull the weight from the servers
    auto val = new std::vector<Real>();
    auto val_siz = new std::vector<int>();

    // this callback will be called when the weight has been actually pulled back
    pull_w_opt.callback = [this, data, feaid, val, val_siz, train]() {
      double start = GetTime();
      // eval the objective, and report progress to the scheduler
      Objective obj(data->GetBlock(), *val, *val_siz, dims_, nt_);
      Progress prog;
      obj.Evaluate(&prog);
      Report(&prog);

      if (train) {
        // calculate and push the gradients
        obj.CalcGrad(val);

        ps::SyncOpts push_grad_opt;
        // filters to reduce network traffic
        SetFilters(2, &push_grad_opt);
        // this callback will be called when the gradients have been actually pushed
        // LL << DebugStr(*val);
        push_grad_opt.callback = [this]() { FinishMinibatch(); };
        server_.ZVPush(feaid,
                       std::shared_ptr<std::vector<Real>>(val),
                       std::shared_ptr<std::vector<int>>(val_siz),
                       push_grad_opt);

      } else {
        FinishMinibatch();
        delete val;
        delete val_siz;
      }
      delete data;
      workload_time_ += GetTime() - start;
    };

    // filters to reduce network traffic
    SetFilters(1, &pull_w_opt);
    server_.ZVPull(feaid, val, val_siz, pull_w_opt);
  }

 private:
  // flag: 0 push feature count, 1 pull weight, 2 push gradient
  void SetFilters(int flag, ps::SyncOpts* opts) {
    if (conf_.key_cache()) {
      opts->AddFilter(ps::Filter::KEY_CACHING)->set_clear_cache(flag == 2);
    }
    if (conf_.fixed_bytes() > 0) {
      if (flag == 0) {
        // trancate the count to uint8
        opts->AddFilter(ps::Filter::TRUNCATE_FLOAT)->set_num_bytes(1);
      } else if (flag == 2) {
        // randomly round the gradient
        opts->AddFilter(ps::Filter::FIXING_FLOAT)->set_num_bytes(
            conf_.fixed_bytes());
      }
    }
    if (conf_.msg_compression()) {
      opts->AddFilter(ps::Filter::COMPRESSING);
    }
  }
  Config conf_;
  ps::KVWorker<Real> server_;

  int nt_ = 2;
  std::vector<int> dims_;
};

}  // namespace fm
}  // namespace dmlc

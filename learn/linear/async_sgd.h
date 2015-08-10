/**
 * @file   async_sgd.h
 * @brief  Asynchronous stochastic gradient descent to solve linear methods.
 */
#include "solver/minibatch_solver.h"
#include "config.pb.h"
#include "progress.h"
#include "base/localizer.h"
#include "loss.h"
#include "penalty.h"

namespace dmlc {
namespace linear {

using FeaID = ps::Key;
template <typename T> using Blob = ps::Blob<T>;

/**
 * \brief the base sgd handle
 */
struct ISGDHandle {
 public:
  ISGDHandle() { ns_ = ps::NodeInfo::NumServers();
    LOG(ERROR) << ns_;
  }
  inline void Start(bool push, int timestamp, int cmd, void* msg) { }

  inline void Finish() {
    // avoid too frequently reporting
    ++ ct_;
    if (ct_ >= ns_ && reporter) {
      Progress prog; prog.new_w() = new_w; reporter(prog);
      new_w = 0; ct_ = 0;
    }
  }

  inline static void Update(float cur_w, float old_w) {
    if (old_w == 0 && cur_w != 0) {
      ++ new_w;
    } else if (old_w != 0 && cur_w == 0) {
      -- new_w;
    }
  }

  void Load(Stream* fi) { }
  void Save(Stream *fo) const { }

  L1L2<float> penalty;

  // learning rate
  float alpha = 0.1, beta = 1;

  std::function<void(const Progress& prog)> reporter;
  static int64_t new_w;

 private:
  int ct_ = 0;
  int ns_ = 0;
};

template <typename T> inline void TLoad(Stream* fi, T* ptr) {
  CHECK_EQ(fi->Read(&ptr->w, sizeof(float)), sizeof(float));
  ISGDHandle::Update(ptr->w, 0);
}

template <typename T> inline void TSave(Stream* fo, T* const ptr) {
  fo->Write(&ptr->w, sizeof(float));
}

/**
 * \brief Standard SGD value
 */
struct SGDEntry {
  float w = 0;
  inline void Load(Stream *fi) { TLoad(fi, this); }
  inline void Save(Stream *fo) const { TSave(fo, this); }
  inline bool Empty() const { return w == 0; }
};

/**
 * \brief Standard SGD handle
 *
 * use alpha / ( beta + sqrt(t)) as the learning rate
 */
struct SGDHandle : public ISGDHandle {
 public:
  inline void Start(bool push, int timestamp, int cmd, void* msg) {
    if (push) {
      eta = (this->beta + sqrt((float)t)) / this->alpha;
      t += 1;
    }
  }
  inline void Push(FeaID key, Blob<const float> grad, SGDEntry& w) {
    float old_w = w.w;
    w.w = penalty.Solve(eta * w.w - grad[0], eta);
    Update(w.w, old_w);
  }

  inline void Pull(FeaID key, const SGDEntry& w, Blob<float>& send) {
    send[0] = w.w;
  }
  int t = 1;
  float eta = 0;
};

/**
 * \brief AdaGrad SGD value.
 */
struct AdaGradEntry {
  float w = 0;
  float sq_cum_grad = 0;  // sqrt(sum_t grad_t^2)

  inline void Load(Stream *fi) { TLoad(fi, this); }
  inline void Save(Stream *fo) const { TSave(fo, this); }
  inline bool Empty() const { return w == 0; }
};


/**
 * \brief AdaGrad SGD handle.
 *
 * use alpha / ( beta + sqrt(sum_t grad_t^2)) as the learning rate
 */
struct AdaGradHandle : public ISGDHandle {
  inline void Push(FeaID key, Blob<const float> grad, AdaGradEntry& val) {
    // update cum grad
    float g = grad[0];
    float sqrt_n = val.sq_cum_grad;
    val.sq_cum_grad = sqrt(sqrt_n * sqrt_n + g * g);

    // update w
    float eta = (val.sq_cum_grad + beta) / alpha;
    float old_w = val.w;
    val.w = penalty.Solve(eta * old_w - g, eta);

    Update(val.w, old_w);
  }

  inline void Pull(FeaID key, const AdaGradEntry& val, Blob<float>& send) {
    send[0] = val.w;
  }
};

/**
 * \brief FTRL value
 */
struct FTRLEntry {
  float w = 0;  // weight
  float z = 0;  // the smoothed version of - eta * w + grad
  float sq_cum_grad = 0; // sqrt(sum_t grad_t^2)

  inline void Load(Stream *fi) { TLoad(fi, this); }
  inline void Save(Stream *fo) const { TSave(fo, this); }
  inline bool Empty() const { return w == 0; }
};

/**
 * \brief FTRL updater, use a smoothed weight for better spasity
 */
struct FTRLHandle : public ISGDHandle {
 public:
  inline void Push(FeaID key, Blob<const float> grad, FTRLEntry& val) {
    // update cum grad
    float g = grad[0];
    float sqrt_n = val.sq_cum_grad;
    val.sq_cum_grad = sqrt(sqrt_n * sqrt_n + g * g);

    // update z
    float old_w = val.w;
    float sigma = (val.sq_cum_grad - sqrt_n) / alpha;
    val.z += g - sigma * old_w;

    // update w
    val.w = penalty.Solve(-val.z, (beta + val.sq_cum_grad) / alpha);

    Update(val.w, old_w);
  }

  inline void Pull(FeaID key, const FTRLEntry& val, Blob<float>& send) {
    send[0] = val.w;
  }
};


class AsgdServer : public solver::MinibatchServer {
 public:
  AsgdServer(const Config& conf) : conf_(conf) {
    auto algo = conf_.algo();
    if (algo == Config::SGD) {
      CreateServer<SGDEntry, SGDHandle>();
    } else if (algo == Config::ADAGRAD) {
      CreateServer<AdaGradEntry, AdaGradHandle>();
    } else if (algo == Config::FTRL) {
      CreateServer<FTRLEntry, FTRLHandle>();
    } else {
      LOG(FATAL) << "unknown algo: " << algo;
    }
  }
  virtual ~AsgdServer() { }

 protected:
  template <typename Entry, typename Handle>
  void CreateServer() {
    Handle h;
    h.penalty.set_lambda1(conf_.lambda_l1());
    h.penalty.set_lambda2(conf_.lambda_l2());
    if (conf_.has_lr_eta()) h.alpha = conf_.lr_eta();
    if (conf_.has_lr_beta()) h.beta = conf_.lr_beta();

    h.reporter = [this](const Progress& prog) {
      ReportToScheduler(prog.data);
    };
    ps::OnlineServer<float, Entry, Handle> s(h);
    server_ = s.server();
  }

  virtual void LoadModel(Stream* fi) {
    server_->Load(fi);
    Progress prog; prog.new_w() = ISGDHandle::new_w; ReportToScheduler(prog.data);
    ISGDHandle::new_w = 0;
  }

  virtual void SaveModel(Stream* fo) const {
    server_->Save(fo);
  }

  Config conf_;
  ps::KVStore* server_;
};

class AsgdWorker : public solver::MinibatchWorker {
 public:
  AsgdWorker(const Config& conf) : conf_(conf) {
    mb_size_       = conf_.minibatch();
    shuffle_       = conf_.rand_shuffle();
    concurrent_mb_ = conf_.max_concurrency();
    neg_sampling_  = conf_.neg_sampling();
  }
  virtual ~AsgdWorker() { }

 protected:
  virtual void ProcessMinibatch(const Minibatch& mb, const Workload& wl) {
    // find the unique feature ids in this minibatch
    auto data = new dmlc::data::RowBlockContainer<unsigned>();
    auto feaid = std::make_shared<std::vector<FeaID>>();

    double start = GetTime();
    Localizer<FeaID> lc(nt_);
    lc.Localize(mb, data, feaid.get());
    workload_time_ += GetTime() - start;

    // pull the weight from the servers
    auto val = new std::vector<float>();
    ps::SyncOpts pull_w_opt;

    // this callback will be called when the weight has been actually pulled
    // back
    int k = wl.file[0].k;
    pull_w_opt.callback = [this, data, feaid, val, k, wl]() {
      double start = GetTime();
      // eval the objective, and report progress to the scheduler
      auto loss = CreateLoss<float>(conf_.loss());
      loss->Init(data->GetBlock(), *val, nt_);

      if (wl.type == Workload::PRED) {
        loss->Predict(PredictStream(conf_.predict_out(), wl), conf_.prob_predict());
      } else {
        Progress prog; loss->Evaluate(&prog); ReportToScheduler(prog.data);
      }

      bool train = wl.type == Workload::TRAIN;
      if (train) {
        // calculate and push the gradients
        loss->CalcGrad(val);

        ps::SyncOpts push_grad_opt;
        // filters to reduce network traffic
        SetFilters(train, &push_grad_opt);
        // this callback will be called when the gradients have been actually
        // pushed
        push_grad_opt.callback = [this]() { FinishMinibatch(); };
        kv_.ZPush(
            feaid, std::shared_ptr<std::vector<float>>(val), push_grad_opt);
      } else {
        FinishMinibatch();
        delete val;
      }
      delete loss;
      delete data;
      workload_time_ += GetTime() - start;
    };
    kv_.ZPull(feaid, val, pull_w_opt);
  }
 private:
  void SetFilters(bool push, ps::SyncOpts* opts) {
    if (conf_.fixed_bytes() > 0) {
      opts->AddFilter(ps::Filter::FIXING_FLOAT)->set_num_bytes(
          conf_.fixed_bytes());
    }
    if (conf_.key_cache()) {
      opts->AddFilter(ps::Filter::KEY_CACHING)->set_clear_cache(push);
    }
    if (conf_.msg_compression()) {
      opts->AddFilter(ps::Filter::COMPRESSING);
    }
  }
  Config conf_;
  int nt_ = 2;
  ps::KVWorker<float> kv_;
};


class AsgdScheduler : public solver::MinibatchScheduler {
 public:
  AsgdScheduler(const Config& conf) { Init(conf); }
  virtual ~AsgdScheduler() { }

  virtual std::string ProgHeader() { return Progress::HeadStr(); }

  virtual std::string ProgString(const solver::Progress& prog) {
    prog_.data = prog;
    return prog_.PrintStr();
  }
 private:
  Progress prog_;
};

}  // namespace linear
}  // namespace dmlc

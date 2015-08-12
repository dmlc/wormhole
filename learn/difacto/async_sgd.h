#pragma once
#include "progress.h"
#include "config.pb.h"
#include "loss.h"
#include "base/localizer.h"
#include "solver/minibatch_solver.h"

namespace dmlc {
namespace difacto {

/**
 * \brief the scheduler for async SGD
 */
class AsyncScheduler : public solver::MinibatchScheduler {
 public:
  AsyncScheduler(const Config& conf) : conf_(conf) {
    if (conf_.early_stop()) {
      CHECK(conf_.val_data().size()) << "early stop needs validation dataset";
    }
    Init(conf);
  }
  virtual ~AsyncScheduler() { }

  virtual std::string ProgHeader() { return Progress::HeadStr(); }

  virtual std::string ProgString(const solver::Progress& prog) {
    prog_.data = prog;
    return prog_.PrintStr();
  }

  virtual bool Stop(const Progress& cur, bool train) {
    double cur_objv = cur.objv() / cur.new_ex();
    if (train) {
      if (conf_.has_max_objv() && cur_objv > conf_.max_objv()) {
        return true;
      }
    } else {
      double diff = pre_val_objv_ - cur_objv;
      pre_val_objv_ = cur_objv;
      if (conf_.early_stop() && diff < conf_.min_objv_decr()) {
        std::cout << "The decrease of validation objective "
                  << "is smaller than the minimal requirement: "
                  << diff << " vs " << conf_.min_objv_decr()
                  << std::endl;
        return true;
      }
    }
    return false;
  }

 private:
  Progress prog_;
  Config conf_;
  double pre_val_objv_ = 100;
};

using FeaID = ps::Key;
template <typename T> using Blob = ps::Blob<T>;
static const int kPushFeaCnt = 1;

/**
 * \brief the base sgd handle
 */
struct ISGDHandle {
  ISGDHandle() { ns_ = ps::NodeInfo::NumServers(); }
  inline void Start(bool push, int timestamp, int cmd, void* msg) {
    push_count = (push && (cmd == kPushFeaCnt)) ? true : false;
    perf_.Start(push, cmd);
  }

  inline void Report() {
    // reduce communication frequency
    ++ ct_;
    if (ct_ >= ns_ && reporter) {
      Progress prog; prog.new_w() = new_w; prog.new_V() = new_V; reporter(prog);
      new_w = 0; new_V = 0;ct_ = 0;
    }
  }

  inline void Finish() { Report(); perf_.Stop(); }

  // for w
  float lambda_l1 = 0, lambda_l2 = 0;
  float alpha = .01, beta = 1;

  // for V
  struct Embedding {
    int dim = 0;
    unsigned thr;
    float lambda_l1 = 0, lambda_l2 = 0;
    float alpha = .01, beta = 1;
    float V_min = -.01, V_max = .01;
  };
  Embedding V;
  bool l1_shrk;

  // statistic
  bool push_count;
  static int64_t new_w;
  static int64_t new_V;
  std::function<void(const Progress& prog)> reporter;

  void Load(Stream* fi) { }
  void Save(Stream *fo) const { }

 private:
  // performance monitor and logger
  class Perf {
   public:
    void Start(bool push, int cmd) {
      time_[0] = GetTime();
      i_ = push ? ((cmd == kPushFeaCnt) ? 1 : 2) : 3;
    }
    void Stop() {
      time_[i_] += GetTime() - time_[0];
      ++ count_[i_]; ++ count_[0];
      if ((count_[0] % disp_) == 0) {
        LOG(INFO) << "push feacnt: " << count_[1] << " x " << time_[1]/count_[1]
                  << ", push grad: " << count_[2] << " x " << time_[2]/count_[2]
                  << ", pull: " << count_[3] << " x " << time_[3]/count_[3];
      }
    }
   private:
    std::array<double, 4> time_{};
    std::array<int, 4> count_{};
    int i_ = 0, disp_ = ps::NodeInfo::NumWorkers() * 10;
  } perf_;

  int ct_ = 0, ns_ = 0;
};

/**
 * \brief value stored on server nodes
 */
struct AdaGradEntry {
  AdaGradEntry() { }
  ~AdaGradEntry() { Clear(); }

  inline void Clear() {
    if ( size > 1 ) { delete [] w; delete [] sqc_grad; }
    size = 0; w = NULL; sqc_grad = NULL;
  }

  inline void Resize(int n) {
    if (n < size) { size = n; return; }

    float* new_w = new float[n]; float* new_cg = new float[n+1];
    if (size == 1) {
      new_w[0] = w_0(); new_cg[0] = sqc_grad_0(); new_cg[1] = z_0();
    } else {
      memcpy(new_w, w, size * sizeof(float));
      memcpy(new_cg, sqc_grad, (size+1) * sizeof(float));
      Clear();
    }
    w = new_w; sqc_grad = new_cg; size = n;
  }

  inline float& w_0() { return size == 1 ? *(float *)&w : w[0]; }
  inline float w_0() const { return size == 1 ? *(float *)&w : w[0]; }

  inline float& sqc_grad_0() {
    return size == 1 ? *(float *)&sqc_grad : sqc_grad[0];
  }

  inline float& z_0() {
    return size == 1 ? *(((float *)&sqc_grad)+1) : sqc_grad[1];
  }

  void Load(Stream* fi) {
    fi->Read(&size, sizeof(size)) ;
    if (size == 1) {
      fi->Read(&w, sizeof(float*));
      fi->Read(&sqc_grad, sizeof(float*));
    } else {
      w = new float[size];
      sqc_grad = new float[size+1];
      fi->Read(w, sizeof(float)*size);
      fi->Read(sqc_grad, sizeof(float)*(size+1));
      ISGDHandle::new_V += size - 1;
    }
    if (w_0() != 0) ++ ISGDHandle::new_w;
  }

  void Save(Stream *fo) const {
    fo->Write(&size, sizeof(size));
    if (size == 1) {
      fo->Write(&w, sizeof(float*));
      fo->Write(&sqc_grad, sizeof(float*));
    } else {
      fo->Write(w, sizeof(float)*size);
      fo->Write(sqc_grad, sizeof(float)*(size+1));
    }
  }

  bool Empty() const { return (w_0() == 0 && size == 1); }

  /// #appearence of this feature in the data
  unsigned fea_cnt = 0;

  /// length of w. if size == 1, then using w itself to store the value to save
  /// memory and avoid unnecessary new (see w_0())
  int size = 1;

  /// w and V
  float *w = NULL;

  /// square root of the cumulative gradient
  float *sqc_grad = NULL;
};

/**
 * \brief model updater
 */
struct AdaGradHandle : public ISGDHandle {

  inline void Push(FeaID key, Blob<const float> recv, AdaGradEntry& val) {
    if (push_count) {
      val.fea_cnt += (unsigned) recv[0];
      Resize(val);
    } else {
      CHECK_LE(recv.size, (size_t)val.size);
      CHECK_GE(recv.size, (size_t)0);

      // update w
      UpdateW(val, recv[0]);

      // update V
      if (recv.size > 1) {
        UpdateV(val.w+1, val.sqc_grad+2, recv.data+1, recv.size-1);
      }
    }
  }

  inline void Pull(FeaID key, const AdaGradEntry& val, Blob<float>& send) {
    float w0 = val.w_0();
    if (val.size == 1 || (l1_shrk && (w0 == 0))) {
      CHECK_GT(send.size, (size_t)0);
      send[0] = w0;
      send.size = 1;
    } else {
      send.data = val.w;
      send.size = val.size;
    }
  }

  /// \brief resize if necessary
  inline void Resize(AdaGradEntry& val) {
    // resize the larger dim first to avoid double resize
    if (val.fea_cnt > V.thr && val.size < V.dim + 1 &&
        (!l1_shrk || val.w_0() != 0)) {
      int old_siz = val.size;
      val.Resize(V.dim + 1);
      for (int j = old_siz; j < val.size; ++j) {
        val.w[j] = rand() / (float) RAND_MAX * (V.V_max - V.V_min) + V.V_min;
        val.sqc_grad[j+1] = 0;
      }
      new_V += val.size - old_siz;
    }
  }

  // ftrl
  inline void UpdateW(AdaGradEntry& val, float g) {
    float w = val.w_0();
    g += lambda_l2 * w;

    float cg = val.sqc_grad_0();
    float cg_new = sqrt( cg * cg + g * g );
    val.sqc_grad_0() = cg_new;

    val.z_0() -= g - (cg_new - cg) / alpha * w;

    float z = val.z_0();
    float l1 = lambda_l1;
    if (z <= l1  && z >= - l1) {
      val.w_0() = 0;
    } else {
      float eta = (beta + cg_new) / alpha;
      val.w_0() = (z > 0 ? z - l1 : z + l1) / eta;
    }

    if (w == 0 && val.w_0() != 0) {
      ++ new_w; Resize(val);
    } else if (w != 0 && val.w_0() == 0) {
      -- new_w;
    }
  }

  // adagrad
  inline void UpdateV(float* w, float* cg, float const* g, int n) {
    for (int i = 0; i < n; ++i) {
      float grad = g[i] + V.lambda_l2 * w[i];
      cg[i] = sqrt(cg[i] * cg[i] + grad * grad);
      float eta = V.alpha / ( cg[i] + V.beta );
      w[i] -= eta * grad;
    }
  }
};

class AsyncServer : public solver::MinibatchServer {
 public:
  AsyncServer(const Config& conf) : conf_(conf) {
    using Server = ps::OnlineServer<float, AdaGradEntry, AdaGradHandle>;
    AdaGradHandle h;
    h.reporter = [this](const Progress& prog) { ReportToScheduler(prog.data); };

    // for w
    h.alpha     = conf.lr_eta();
    h.beta      = conf.lr_beta();
    h.lambda_l1 = conf.lambda_l1();
    h.lambda_l2 = conf.lambda_l2();
    h.l1_shrk   = conf.l1_shrk();

    // for V
    if (conf.embedding_size() > 0) {
      const auto& c = conf.embedding(0);
      h.V.dim       = c.dim();
      h.V.thr       = (unsigned)c.threshold();
      h.V.lambda_l2 = c.lambda_l2();
      h.V.V_min     = - c.init_scale();
      h.V.V_max     = c.init_scale();
      h.V.alpha     = c.has_lr_eta() ? c.lr_eta() : h.alpha;
      h.V.beta      = c.has_lr_beta() ? c.lr_beta() : h.beta;
    }

    Server s(h);
    server_ = s.server();
  }

  virtual ~AsyncServer() { }
 protected:
  virtual void LoadModel(Stream* fi) {
    server_->Load(fi);

    Progress prog;
    prog.new_w() = ISGDHandle::new_w; prog.new_V() = ISGDHandle::new_V;
    ReportToScheduler(prog.data);
  }

  virtual void SaveModel(Stream* fo) const {
    server_->Save(fo);
  }
  ps::KVStore* server_;
  Config conf_;
};

class AsyncWorker : public solver::MinibatchWorker {
 public:
  AsyncWorker(const Config& conf) : conf_(conf) {
    mb_size_       = conf_.minibatch();
    shuffle_       = conf_.rand_shuffle();
    concurrent_mb_ = conf_.max_concurrency();
    neg_sampling_  = conf_.neg_sampling();
    for (int i = 0; i < conf.embedding_size(); ++i) {
      if (conf.embedding(i).dim() > 0) {
        do_embedding_ = true; break;
      }
    }
  }
  virtual ~AsyncWorker() { }

 protected:

  virtual void ProcessMinibatch(const Minibatch& mb, const Workload& wl) {
    auto data = new dmlc::data::RowBlockContainer<unsigned>();
    auto feaid = std::make_shared<std::vector<FeaID>>();
    auto feacnt = std::make_shared<std::vector<float>>();

    double start = GetTime();
    Localizer<FeaID> lc(conf_.num_threads());
    lc.Localize(mb, data, feaid.get(), feacnt.get());
    workload_time_ += GetTime() - start;

    ps::SyncOpts pull_w_opt;
    if (wl.type == Workload::TRAIN && wl.data_pass == 0 && do_embedding_) {
      // push the feature count to the servers
      ps::SyncOpts cnt_opt;
      SetFilters(0, &cnt_opt);
      cnt_opt.cmd = kPushFeaCnt;
      int t = server_.ZPush(feaid, feacnt, cnt_opt);
      pull_w_opt.deps.push_back(t);
      // LL << DebugStr(*feacnt);
    }

    // pull the weight from the servers
    auto val = new std::vector<float>();
    auto val_siz = new std::vector<int>();

    // this callback will be called when the weight has been actually pulled
    // back
    pull_w_opt.callback = [this, data, feaid, val, val_siz, wl]() {
      double start = GetTime();
      // eval the objective, and report progress to the scheduler
      Loss<float> loss(data->GetBlock(), *val, *val_siz, conf_);
      if (wl.type == Workload::PRED) {
        loss.Predict(PredictStream(conf_.predict_out(), wl), conf_.prob_predict());
      } else {
        Progress prog; loss.Evaluate(&prog); ReportToScheduler(prog.data);
      }

      if (wl.type == Workload::TRAIN) {
        // calculate and push the gradients
        loss.CalcGrad(val);

        ps::SyncOpts push_grad_opt;
        // filters to reduce network traffic
        SetFilters(2, &push_grad_opt);
        // this callback will be called when the gradients have been actually
        // pushed
        // LL << DebugStr(*val);
        push_grad_opt.callback = [this]() { FinishMinibatch(); };
        server_.ZVPush(feaid,
                       std::shared_ptr<std::vector<float>>(val),
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
      } else {
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
  bool do_embedding_ = false;
  ps::KVWorker<float> server_;
};


}  // namespace difacto
}  // namespace dmlc

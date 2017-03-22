/* 
 * File:   async_sgd.h
 * Author: hexi
 *
 * Created on 2015年12月26日, 下午1:55
 */

#pragma once
#include "progress.h"
#include "config.pb.h"
#include "loss.h"
#include "localizer.h"
#include "solver/minibatch_solver.h"

namespace dmlc {
namespace svdfeature {

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
static const int kPushFeaType = 1;

/**
 * \brief the base sgd handle
 */
struct ISGDHandle {
  ISGDHandle() { ns_ = ps::NodeInfo::NumServers(); }
  inline void Start(bool push, int timestamp, int cmd, void* msg) {
    push_ft = (push && (cmd == kPushFeaType)) ? true : false;
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
  struct MF {
    int dim = 0;
    float lambda_l1 = 0, lambda_l2 = 0;
    float alpha = .01, beta = 1;
    float V_min = -.01, V_max = .01;
    bool no_user_bias = false;
  };
  MF V;

  // statistic
  bool push_ft;
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
      i_ = push ? ((cmd == kPushFeaType) ? 1 : 2) : 3;
    }
    void Stop() {
      time_[i_] += GetTime() - time_[0];
      ++ count_[i_]; ++ count_[0];
      if ((count_[0] % disp_) == 0) {
        LOG(INFO) << "push featype: " << count_[1] << " x " << time_[1]/count_[1]
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
    if (n <= size) { size = n; return; }

    float* new_w = new float[n]; float* new_cg = new float[n];
    if (size == 1) {
      new_w[0] = w_0(); new_cg[0] = sqc_grad_0();
    } else {
      memcpy(new_w, w, size * sizeof(float));
      memcpy(new_cg, sqc_grad, size * sizeof(float));
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
    CHECK_EQ(size, size_t(1));
    return *(((float *)&sqc_grad)+1);
  }

  void Load(Stream* fi) {
    fi->Read(&feat_type, sizeof(int));  
    fi->Read(&size, sizeof(int)) ;
    if (size == 1) {
      fi->Read(&w, sizeof(float*));
      fi->Read(&sqc_grad, sizeof(float*));
      if (w_0() != 0) ++ ISGDHandle::new_w;
    } else {
      w = new float[size];
      sqc_grad = new float[size];
      fi->Read(w, sizeof(float)*size);
      fi->Read(sqc_grad, sizeof(float)*size);
      ISGDHandle::new_V += size;
    }
  }

  void Save(Stream *fo) const {
    fo->Write(&feat_type, sizeof(int));
    fo->Write(&size, sizeof(int));
    if (size == 1) {
      fo->Write(&w, sizeof(float*));
      fo->Write(&sqc_grad, sizeof(float*));
    } else {
      fo->Write(w, sizeof(float)*size);
      fo->Write(sqc_grad, sizeof(float)*size);
    }
  }

  bool Empty() const { return (w_0() == 0 && size == 1); }

  /// #appearence of this feature in the data
  int feat_type = -1;

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
      
      if(push_ft) {
         if(val.feat_type < 0) {
            val.feat_type = (int)recv[0];
            Resize(val);
         }else{
            CHECK_EQ((int)recv[0], val.feat_type);
         } 
      } else {
        CHECK_LE(recv.size, (size_t)val.size);
        CHECK_GE(recv.size, (size_t)0);
        if(recv.size > 1) {
            // feat_type => 0:user, 1:item, 2: global
            CHECK(val.feat_type == 0 || val.feat_type == 1 );
            UpdateV(val.w, val.sqc_grad, recv.data, recv.size);
            
        } else {
            CHECK_EQ(recv.size, (size_t)1);
            UpdateW(val, recv[0]);
        }
        // feat_type = 0 -> user vector
        if(V.no_user_bias && val.feat_type == 0) {
            val.w_0() = 0;
            val.sqc_grad_0() = 0;
        }
     }
  }

  inline void Pull(FeaID key, AdaGradEntry& val, Blob<float>& send) {
    float w0 = val.w_0();
    if (val.size == 1) {
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
    // create vector if ness
    // feat_type: 0:user, 1:item, 2:global
    if (val.feat_type < 2 && val.size < V.dim + 1) {
      int old_siz = val.size;
      val.Resize(V.dim + 1);
      for (int j = old_siz; j < val.size; ++j) {
        val.w[j] = rand() / (float) RAND_MAX * (V.V_max - V.V_min) + V.V_min;
        val.sqc_grad[j] = 0;
      }
      new_V += val.size - old_siz;
    }
  }

  // ftrl
  inline void UpdateW(AdaGradEntry& val, float g) {
    float w = val.w_0();
    float cg = val.sqc_grad_0();
    float cg_new = sqrt( cg * cg + g * g );
    val.sqc_grad_0() = cg_new;

    val.z_0() += g - (cg_new - cg) / alpha * w;

    float z = val.z_0();
    
    float l1 = lambda_l1;
    if (z <= l1  && z >= - l1) {
      val.w_0() = 0;
    } else {
      float eta = (beta + cg_new) / alpha;
      val.w_0() = -1.0f * (z > 0 ? z - l1 : z + l1) / (eta + lambda_l2);
    }

    if (w == 0 && val.w_0() != 0) {
      ++ new_w;
    } else if (w != 0 && val.w_0() == 0) {
      -- new_w;
    }
  }

  // adagrad
  inline void UpdateV(float* w, float* cg, float const* g, int n) {
    float old_w = w[0];
    for (int i = 0; i < n; ++i) {
      float grad = g[i] + V.lambda_l2 * w[i];
      cg[i] = sqrt(cg[i] * cg[i] + grad * grad);
      float eta = V.alpha / ( cg[i] + V.beta );
      w[i] -= eta * grad;
    }
    // update |w| count
    if (old_w == 0 && w[0] != 0) {
      ++ new_w;
    } else if (old_w != 0 && w[0] == 0) {
      -- new_w;
    }
  }
};

class AsyncServer : public solver::MinibatchServer {
 public:
  AsyncServer(const Config& conf) : conf_(conf) {
    using Server = ps::OnlineServer<float, AdaGradEntry, AdaGradHandle>;
    AdaGradHandle h;
    h.reporter = [this](const Progress& prog) { ReportToScheduler(prog.data); };

    // for global bias: w
    h.alpha     = conf.lr_eta();
    h.beta      = conf.lr_beta();
    h.lambda_l1 = conf.lambda_l1();
    h.lambda_l2 = conf.lambda_l2();

    // for V
    if (conf.mf_size() > 0) {
      const auto& c = conf.mf(0);
      h.V.dim       = c.dim();
      h.V.lambda_l2 = c.lambda_l2();
      h.V.V_min     = - c.init_scale();
      h.V.V_max     = c.init_scale();
      h.V.alpha     = c.has_lr_eta() ? c.lr_eta() : h.alpha;
      h.V.beta      = c.has_lr_beta() ? c.lr_beta() : h.beta;
      h.V.no_user_bias = c.no_user_bias();
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
    nt_ = conf.num_threads();
    for (int i = 0; i < conf.mf_size(); ++i) {
      if (conf.mf(i).dim() > 0) {
        do_mf_ = true; 
        dim_ = conf.mf(i).dim();
        break;
      }
    }
  }
  virtual ~AsyncWorker() { }

 protected:

  virtual void ProcessMinibatch(const Minibatch& mb, const Workload& wl) {
    
    auto data = new dmlc::data::RowBlockContainer<unsigned>();
    auto feaid = std::make_shared<std::vector<FeaID>>();
    auto featype = std::make_shared<std::vector<float>>();
    
    double start = GetTime();
    Localizer<FeaID> lc(conf_.num_threads());
        
    lc.Localize(mb, data, feaid.get());
    
    workload_time_ += GetTime() - start;
    
    ps::SyncOpts pull_w_opt;
    
    featype->resize(feaid.get()->size());
    for(size_t i = 0; i < feaid->size(); ++i) {
        (*featype)[i] = (float)((*feaid)[i] & 3);
    }
    
    if (wl.type == Workload::TRAIN && wl.data_pass == 0 && do_mf_) {
      // push the feature count to the servers
      ps::SyncOpts ft_opt;
      SetFilters(0, &ft_opt);
      ft_opt.cmd = kPushFeaType;
      int t = server_.ZPush(feaid, featype, ft_opt);
      pull_w_opt.deps.push_back(t);
      // LL << DebugStr(*feacnt);
    }

    // pull the weight from the servers
    auto val = new std::vector<float>();
    auto val_siz = new std::vector<int>();

    // this callback will be called when the weight has been actually pulled
    // back
    pull_w_opt.callback = [this, data, feaid, featype, val, val_siz, wl]() {
        
      double start = GetTime();
      // eval the objective, and report progress to the scheduler
      auto loss = CreateLoss<float>(conf_.loss());
      loss->UseBias(conf_.bias());
      loss->Init(data->GetBlock(), *featype, *val, *val_siz, nt_, dim_);   
      Progress prog; loss->Evaluate(&prog); ReportToScheduler(prog.data);
      
      if (wl.type == Workload::PRED) {
        loss->Predict(PredictStream(conf_.predict_out(), wl), conf_.prob_predict());
      } 
      if (wl.type == Workload::TRAIN) {
        
        // calculate and push the gradients
        loss->CalcGrad(val);

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
      delete loss;
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
  bool do_mf_ = false;
  int dim_ = 0;
  int nt_ = 2;
  ps::KVWorker<float> server_;
};


}  // namespace difacto
}  // namespace dmlc



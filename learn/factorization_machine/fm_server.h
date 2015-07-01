#pragma once
#include "fm.h"

namespace dmlc {
namespace fm {
template <typename T> using Blob = ps::Blob<T>;

////////////////////////////////////////////////////////////
//   model
////////////////////////////////////////////////////////////

struct AdaGradEntry {
  AdaGradEntry() { }
  ~AdaGradEntry() { Clear(); }

  inline void Clear() {
    if ( size > 1 ) { delete [] w; delete [] sq_cum_grad; }
    size = 0;
  }

  inline void Resize(int n) {
    if (n < size) { size = n; return; }

    Real* new_w = new Real[n];
    Real* new_cg = new Real[n+1];

    if (size == 1) {
      new_w[0] = w_0();
      new_cg[0] = sq_cum_grad_0();
      new_cg[1] = z_0();
    } else {
      memcpy(new_w, w, size * sizeof(Real));
      memcpy(new_cg, sq_cum_grad, (size+1) * sizeof(Real));
      Clear();
    }

    w = new_w;
    sq_cum_grad = new_cg;
    size = n;
  }

  inline Real& w_0() {
    return size == 1 ? *(Real *)&w : w[0];
  }
  inline Real& sq_cum_grad_0() {
    return size == 1 ? *(Real *)&sq_cum_grad : sq_cum_grad[0];
  }

  inline Real w_0() const {
    return size == 1 ? *(Real *)&w : w[0];
  }

  inline Real& z_0() {
    return size == 1 ? *(((Real *)&sq_cum_grad)+1) : sq_cum_grad[1];
  }

  // appearence of this feature in the data
  unsigned fea_cnt = 0;

  // length of w. if size == 1, then using w itself to store the value to save
  // memory and avoid unnecessary new (see w_0())
  int size = 1;

  Real *w = NULL;
  Real *sq_cum_grad = NULL;
};

////////////////////////////////////////////////////////////
//  model updater
////////////////////////////////////////////////////////////

/**
 * \brief the base handle class
 */
struct ISGDHandle {

  inline void Start(bool push, int timestamp, int cmd, void* msg) {
    push_count = (push && (cmd == kPushFeaCnt)) ? true : false;
    perf.Start(push, cmd);
    // if (push && !push_count) { // for debug
    //   LL << ps::SArray<Real>(((ps::Message*)msg)->value[0]);
    // }
  }

  inline void Finish() {
    if (new_w + new_V > 10000) {
      Progress prog; prog.nnz_w() = new_w; prog.nnz_V() = new_V;
      if (reporter) reporter(prog);
      new_w = 0; new_V = 0;
    }
    perf.Stop();
  }
  inline void SetCaller(void *obj) { }

  bool push_count;

  int64_t new_w = 0;
  int64_t new_V = 0;
  std::function<void(const Progress& prog)> reporter;

  struct Group {
    int dim = 0;
    unsigned thr;
    Real lambda_l1 = 0;
    Real lambda_l2 = 0;
    Real alpha = .01;
    Real beta = 1;
    Real V_min = -.01;
    Real V_max = .01;
  };
  std::array<Group, 3> group;

 private:
  // performance monitor
  class Perf {
   public:
    // Perf() { disp_ = ps::NumWorkers() * 5; }
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
    int i_ = 0;
    int disp_ = ps::NumWorkers() * 5;;
  } perf;
};

struct AdaGradHandle : public ISGDHandle {

  AdaGradHandle() {
    CHECK_EQ(sizeof(Real*), sizeof(Real)*2) << " the reason see z_0()";
  }
  inline void Push(FeaID key, Blob<const Real> recv, AdaGradEntry& val) {
    if (push_count) {
      val.fea_cnt += (unsigned) recv[0];
      // resize the larger dim first to avoid double resize
      for (int i = 2; i > 0; --i) {
        if (val.fea_cnt > group[i].thr &&
            val.size < group[i].dim + 1 &&
            val.w_0() != 0) {
          int old_siz = val.size;
          const auto& g = group[i];
          val.Resize(g.dim + 1);
          for (int j = old_siz; j < val.size; ++j) {
            val.w[j] = rand() / (Real) RAND_MAX * (g.V_max - g.V_min) + g.V_min;
            val.sq_cum_grad[j+1] = 0;
          }
          new_V += val.size - old_siz;
        }
      }
    } else {
      CHECK_LE(recv.size, (size_t)val.size);
      CHECK_GE(recv.size, (size_t)0);

      // update w
      Real old_w = val.w_0();
      UpdateW(val, recv[0]);
      if (old_w == 0 && val.w_0() != 0) {
        ++ new_w;
      } else if (old_w != 0 && val.w_0() == 0) {
        -- new_w;
      }

      // update V
      int rsz = recv.size;
      int pos = 1;
      for (int i = 1; i < 3; ++i) {
        if (rsz <= pos) break;
        UpdateV(val.w + pos,
                val.sq_cum_grad + pos + 1,
                recv.data + pos,
                std::min(rsz - pos, group[i].dim), i);
      }
    }
  }

  inline void Pull(FeaID key, const AdaGradEntry& val, Blob<Real>& send) {
    Real w0 = val.w_0();
    if (val.size == 1 || w0 == 0) {  // trick: don't send V_i if w_i = 0
      CHECK_GT(send.size, (size_t)0);
      send[0] = w0;
      send.size = 1;
    } else {
      send.data = val.w;
      send.size = val.size;
    }
  }

  // ftrl
  inline void UpdateW(AdaGradEntry& val, Real g) {
    auto const& g0 = group[0];

    Real cg = val.sq_cum_grad_0();
    Real cg_new = sqrt( cg * cg + g * g );
    val.sq_cum_grad_0() = cg_new;

    Real w = val.w_0();
    val.z_0() -= g - (cg_new - cg) / g0.alpha * w;

    Real z = val.z_0();
    Real l1 = g0.lambda_l1;
    if (z <= l1  && z >= - l1) {
      val.w_0() = 0;
    } else {
      val.w_0() = (z > 0 ? z - l1 : z + l1) /
                  ((g0.beta + cg_new) / g0.alpha + g0.lambda_l2);
    }
  }

  // adagrad
  inline void UpdateV(Real* w, Real* cg, Real const* g, int n, int d) {
    Real alpha = group[d].alpha;
    Real beta = group[d].beta;
    Real l2 = group[d].lambda_l2;
    for (int i = 0; i < n; ++i) {
      cg[i] = sqrt(cg[i] * cg[i] + g[i] * g[i]);
      Real eta = alpha / ( cg[i] + beta );
      w[i] -= eta * ( g[i] + l2 * w[i] );
    }
  }
};

class FMServer : public solver::AsyncSGDServer {
 public:
  FMServer(const Config& conf) {
    using Server = ps::OnlineServer<AdaGradEntry, Real, AdaGradHandle>;
    AdaGradHandle h;
    h.reporter = [this](const Progress& prog) {
      Report(&prog);
    };

    // for w
    auto& g     = h.group[0];
    g.dim       = 1;
    g.alpha     = conf.lr_eta();
    g.beta      = conf.lr_beta();
    g.lambda_l1 = conf.lambda_l1();
    g.lambda_l2 = conf.lambda_l2();

    // for V
    CHECK_LE(conf.embedding_size(), 2);
    for (int i = 0; i < conf.embedding_size(); ++i) {
      const auto& c = conf.embedding(i);
      auto& g       = h.group[i+1];
      g.dim         = c.dim();
      g.thr         = (unsigned)c.threshold();
      g.alpha       = c.lr_eta();
      g.beta        = c.lr_beta();
      g.lambda_l2   = c.lambda_l2();
      g.V_min       = - c.init_scale();
      g.V_max       = c.init_scale();
    }

    Server s(h);
    server_ = s.server();
  }

  virtual ~FMServer() { }
 protected:
  virtual void SaveModel() { }
  ps::KVStore* server_;
};
}  // namespace fm
}  // namespace dmlc

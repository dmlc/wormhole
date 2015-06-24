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
    Real* new_cg = new Real[n];

    if (size == 1) {
      new_w[0] = w_0();
      new_cg[0] = sq_cum_grad_0();
    } else {
      memcpy(new_w, w, size * sizeof(Real));
      memcpy(new_cg, sq_cum_grad, size * sizeof(Real));
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
  inline Real sq_cum_grad_0() const {
    return size == 1 ? *(Real *)&sq_cum_grad : sq_cum_grad[0];
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
  ISGDHandle() {}

  inline void Start(bool push, int timestamp, int cmd, void* msg) {
    push_count = (push && (cmd == kPushFeaCnt)) ? true : false;
  }
  inline void Finish() {
    if (new_w_entry > 1000) {
      Progress prog; prog.nnz_w() = new_w_entry;
      if (reporter) reporter(prog);
      new_w_entry = 0;
    }
  }
  inline void SetCaller(void *obj) { }

  bool push_count;

  int64_t new_w_entry = 0;
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

};

struct AdaGradHandle : public ISGDHandle {
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
            val.sq_cum_grad[j] = 0;
          }
          new_w_entry += val.size - old_siz;
        }
      }
    } else {
      CHECK_LE(recv.size, val.size);
      CHECK_GE(recv.size, 0);

      // update w
      Real old_w = val.w_0();
      Update(val.w_0(), val.sq_cum_grad_0(), recv[0], 0);
      if (old_w == 0 && val.w_0() != 0) {
        ++ new_w_entry;
      } else if (old_w != 0 && val.w_0() == 0) {
        -- new_w_entry;
      }

      // update V
      if (recv.size > 1) {
        int g = ((int)val.size == group[1].dim + 1) ? 1 : 2;
        for (size_t i = 1; i < recv.size; ++i) {
          Update(val.w[i], val.sq_cum_grad[i], recv[i], g);
        }
      }
    }
  }

  inline void Pull(FeaID key, const AdaGradEntry& val, Blob<Real>& send) {
    if (val.size == 1) {
      CHECK_GT(send.size, 0);
      send[0] = val.w_0();
      send.size = 1;
    } else {
      send.data = val.w;
      send.size = val.size;
    }
  }

  inline void Update(Real& w, Real& cg, Real g, int i) {
    cg = sqrt(cg * cg + g * g);
    Real eta = group[i].alpha / (cg + group[i].beta);
    w = w - eta * ( g + group[i].lambda_l2 * w );
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
      g.V_min       = c.init_min();
      g.V_max       = c.init_max();
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

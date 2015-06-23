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
    Real* new_w = new Real[n]; memset(new_w, 0, sizeof(Real)*n);
    Real* new_cg = new Real[n]; memset(new_cg, 0, sizeof(Real)*n);

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
    return size == 1 ? *(Real *)w : w[0];
  }
  inline Real& sq_cum_grad_0() {
    return size == 1 ? *(Real *)sq_cum_grad : sq_cum_grad[0];
  }

  inline Real w_0() const {
    return size == 1 ? *(Real *)w : w[0];
  }
  inline Real sq_cum_grad_0() const {
    return size == 1 ? *(Real *)sq_cum_grad : sq_cum_grad[0];
  }


  // appearence of this feature in the data
  unsigned fea_cnt = 0;

  // length of w. if size == 1, then using w itself to store the value to save
  // memory and avoid unnecessary new (see w_0())
  unsigned size = 1;

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
  }
  inline void Finish() {
    // report to scheduler

  }
  inline void SetCaller(void *obj) { }

  bool push_count;
  // ModelMonitor* tracker = nullptr;

  Real alpha = 0.1, beta = 1;

  struct Embedding {
    int dim = 0;
    unsigned thr;
    Real lambda = 0;
  };
  std::array<Embedding, 2> embed;

  Real lambda_l1 = 0;
  Real lambda_l2 = 0;
};

struct AdaGradHandle : public ISGDHandle {
  inline void Push(FeaID key, Blob<const Real> recv, AdaGradEntry& val) {
    if (push_count) {
      val.fea_cnt += (unsigned) recv[0];
      // resize 1 first to avoid double resize
      for (int i = 1; i > 0; --i) {
        if (val.fea_cnt > embed[i].thr && (int)val.size < embed[i].dim) {
          val.Resize(embed[i].dim + 1);
        }
      }
    } else {
      CHECK_LE(recv.size, val.size);
      CHECK_GE(recv.size, 0);

      // update w
      Update(val.w_0(), val.sq_cum_grad_0(), recv[0], lambda_l2, lambda_l1);

      // update u
      if (recv.size > 1) {
        Real lam = ((int)val.size == embed[0].dim + 1) ? embed[0].lambda : embed[1].lambda;
        for (size_t i = 1; i < recv.size; ++i) {
          Update(val.w[i], val.sq_cum_grad[i], recv[i], lam, 0);
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

  inline void Update(Real& w, Real& cg, Real g, Real l2, Real l1) {
    cg = sqrt(cg*cg + g*g);
    Real eta = this->alpha / (cg + this->beta);
    w = w - eta * g - l2 * w;
  }
};

class FMServer : public solver::AsyncSGDServer {
 public:
  FMServer(const Config& conf) {
    using Server = ps::OnlineServer<AdaGradEntry, Real, AdaGradHandle>;
    AdaGradHandle h;
    h.alpha     = conf.lr_eta();
    h.beta      = conf.lr_beta();
    h.lambda_l1 = conf.lambda_l1();
    h.lambda_l2 = conf.lambda_l2();

    CHECK_LE(conf.embedding_size(), 2);
    for (int i = 0; i < conf.embedding_size(); ++i) {
      const auto& c     = conf.embedding(i);
      h.embed[i].thr    = (unsigned)c.threshold();
      h.embed[i].dim    = c.dim();
      h.embed[i].lambda = c.lambda_l2();
    }
    Server s(h, Server::kDynamicSize);
    server_ = s.server();
  }

  virtual ~FMServer() { }
 protected:
  virtual void SaveModel() { }
  ps::KVStore* server_;
};
}  // namespace fm
}  // namespace dmlc

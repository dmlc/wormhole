#pragma once
#include "fm.h"

namespace dmlc {
namespace fm {
template <typename T> using Blob = ps::Blob<T>;

////////////////////////////////////////////////////////////
//   model
////////////////////////////////////////////////////////////

// #pragma pack(push)
// #pragma pack(4)
// #pragma pack(pop)

struct AdaGradEntry {
  AdaGradEntry() { }
  ~AdaGradEntry() { if ( size > 1 ) { delete [] w; delete [] sq_cum_grad; } }

  // appearence of this feature in the seened data
  unsigned fea_cnt = 0;

  // length of w. if size == 1, then use w itself to store the value
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
  inline void Finish() {  }
  inline void SetCaller(void *obj) { }

  bool push_count;
  // ModelMonitor* tracker = nullptr;

  Real alpha = 0.1, beta = 1;

  struct Embedding {
    int dim = 0;
    int thr;
    Real lambda = 0;
  };
  std::array<2> embed;

  Real lambda_l1;
  Real lambda_l2;

  // int fea_thr = 100;
  // int fea_thr2 = 1000000;

  // int k = 50;
  // int k2 = 1000;
};

struct AdaGradHandle : public ISGDHandle {
  inline void Push(FeaID key, Blob<const Real> recv, AdaGradEntry& val) {
    if (push_count) {
      val.fea_cnt += (unsigned) recv[0];

      // resize 1 first to avoid double resize
      for (int i = 1; i > 0; --i) {
        if (val.fea_cnt > embed[i].thr && val.size < embed[i].dim) {
          Resize(embed[i].dim + 1, val);
        }
      }
    } else {
      CHECK_LE(recv.size, val.size);
      if (val.size == 1) {
        Update(Cast(&val.w), Cast(&val.sq_cum_grad), recv[0], lambda_l2, lambda_l1);
      } else {
        Update(val.w[0], val.sq_cum_grad[0], recv[0], lambda_l2, lambda_l1);

        Real lam = (val.size == embed[0].dim + 1) ? embed[0].lambda : embed[1].lambda;
        for (size_t i = 1; i < recv.size; ++i) {
          Update(val.w[i], val.sq_cum_grad[i], recv[i], lam, 0);
        }
      }
    }
  }

  inline void Pull(FeaID key, const AdaGradEntry& val, Blob<Real>& send) {
    if (val.size == 1) {
      send[0] = (Real) val.size;
      send.size = 1;
    } else {
      send.data = val.w;
      send.size = val.size;
    }
  }

  inline void Update(Real& w, Real& cg, Real g, Real l2, Real l1) {
    cg = sqrt(cg*cg + g*g);
    Real eta = (cg + this->beta) / this->alpha;
    w = w - eta * g - l2 * w;
  }

  inline Real& Cast(Real** val) { return *(Real *)val; }

  inline void Resize(int n, AdaGradEntry& val) {
    Real* new_w = new Real[n]; memset(new_w, 0, sizeof(Real)*n);
    Real* new_cg = new Real[n]; memset(new_cg, 0, sizeof(Real)*n);

    if (val.size == 1) {
      new_w[0] = Cast(&val.w);
      new_cg[0] = Cast(&val.sq_cum_grad);
    } else {
      memcpy(new_w, val.w, val.size * sizeof(Real));
      memcpy(new_cg, val.sq_cum_grad, val.size * sizeof(Real));
      delete [] val.w;
      delete [] val.sq_cum_grad;
    }

    val.w = new_w;
    val.sq_cum_grad = new_cg;
  }
};

class FMServer : public solver::AsyncSGDServer {
 public:
  FMServer(const Config& conf) {
    using Server = ps::OnlineServer<AdaGradEntry, Real, AdaGradHandle>;
    AdaGradHandle h;
    h.alpha     = lr_eta;
    h.beta      = lr_beta;
    h.lambda_l1 = conf.lambda_l1;
    h.lambda_l2 = conf.lambda_l2;

    CHECK_LE(conf.embedding_size(), 2);
    for (int i = 0; i < conf.embedding_size(); ++i) {
      const auto& c     = conf.embedding(i);
      h.embed[i].thr    = c.threshold();
      h.embed[i].dim    = c.dim();
      h.embed[i].lambda = c.lambda_l2();
    }
    Server s(h, Server::kDynamicSize);
    servers_ = s.server();
  }

  virtual ~FMServer() { }
 protected:
  virtual void SaveModel() { }
  ps::KVStore* server_;
};
}  // namespace fm
}  // namespace dmlc

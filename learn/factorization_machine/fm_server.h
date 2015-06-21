#pragma once
#include "fm.h"

namespace dmlc {
namespace fm {

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
  Real theta = 1;

  std::vector<unsigned> fea_thr;
  std::vector<unsigned> k;

  // int fea_thr = 100;
  // int fea_thr2 = 1000000;

  // int k = 50;
  // int k2 = 1000;
};

struct AdaGradHandle : public ISGDHandle {
  inline void Push(FeaID key, Blob<const Real> recv, AdaGradEntry& val) {
    if (push_count) {
      val.fea_cnt += (unsigned) recv[0];
      for (size_t i = fea_thr.size(); i > 0; --i) {
        if (val.fea_cnt > fea_thr[i-1] && val.size < k[i-1]) {
          Resize(k[i-1] + 1, val); break;
        }
      }
    } else {
      CHECK_LE(recv.size, val.size);
      if (val.size == 1) {
        Update(Cast(&val.w), Cast(&val.sq_cum_grad), recv[0]);
      } else {
        for (size_t i = 0; i < recv.size; ++i) {
          Update(val.w[i], val.sq_cum_grad[i], recv[i]);
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

  inline void Update(Real& w, Real& cg, Real g) {
    cg = sqrt(cg*cg + g*g);
    Real eta = (cg + this->beta) / this->alpha;
    w = w - eta * g;
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
    Server s(h, Server::kDynamicSize);
  }
  virtual ~FMServer() { }
 protected:
  virtual void SaveModel() { }
};
}  // namespace fm
}  // namespace dmlc

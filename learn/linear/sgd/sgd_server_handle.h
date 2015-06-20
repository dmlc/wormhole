/**
 * @file   sgd_server_handle.h
 * @brief  server handles for sgd.
 *
 * No virtual function for better performance
 */
#pragma once
#include "base/monitor.h"
#include "base/penalty.h"
#include "ps/blob.h"
namespace dmlc {
namespace linear {

using FeaID = ps::Key;
using Real = float;
template <typename T> using Blob = ps::Blob<T>;

/*********************************************************************
 * \brief the base handle class
 *********************************************************************/
struct ISGDHandle {
 public:
  inline void Start(bool push, int timestamp, int cmd, void* msg) { }
  inline void Finish() { tracker->Report(); }
  inline void SetCaller(void *obj) { }

  ModelMonitor* tracker = nullptr;
  L1L2<Real> penalty;

  // learning rate
  Real alpha = 0.1, beta = 1;
  Real theta = 1;
};

/*********************************************************************
 * \brief Standard SGD
 * use alpha / ( beta + sqrt(t)) as the learning rate
 *********************************************************************/
struct SGDHandle : public ISGDHandle {
 public:
  inline void Init(FeaID key,  Real& val) { val = 0; }

  inline void Start(bool push, int timestamp, int cmd, void* msg) {
    if (push) {
      eta = (this->beta + sqrt((Real)t)) / this->alpha;
      t += 1;
    }
  }

  inline void Push(FeaID key, Blob<const Real> grad, Real& w) {
    Real old_w = w;
    w = this->penalty.Solve(eta * w - grad[0], eta);
    this->tracker->Update(w, old_w);
  }

  inline void Pull(FeaID key, const Real& w, Blob<Real>& send) {
    send[0] = w;
  }

  // iteration count
  int t = 1;
  Real eta = 0;
};


/*********************************************************************
 * \brief AdaGrad SGD handle.
 * use alpha / ( beta + sqrt(sum_t grad_t^2)) as the learning rate
 *
 * sq_cum_grad: sqrt(sum_t grad_t^2)
 *********************************************************************/

struct AdaGradEntry { Real w = 0; Real sq_cum_grad = 0; };

struct AdaGradHandle : public ISGDHandle {
  inline void Init(FeaID key,  AdaGradEntry& val) { }

  inline void Push(FeaID key, Blob<const Real> grad, AdaGradEntry& val) {
    // update cum grad
    Real g = grad[0];
    Real sqrt_n = val.sq_cum_grad;
    val.sq_cum_grad = sqrt(sqrt_n * sqrt_n + g * g);

    // update w
    Real eta = (val.sq_cum_grad + this->beta) / this->alpha;
    Real old_w = val.w;
    val.w = this->penalty.Solve(eta * old_w - g, eta);

    // update monitor
    this->tracker->Update(val.w, old_w);
  }

  inline void Pull(FeaID key, const AdaGradEntry& val, Blob<Real>& send) {
    send[0] = val.w;
  }
};


/*********************************************************************
 * \brief FTRL updater, use a smoothed weight for better spasity
 *
 * w : weight
 * z : the smoothed version of - eta * w + grad
 * sq_cum_grad: sqrt(sum_t grad_t^2)
 *********************************************************************/

struct FTRLEntry { Real w = 0; Real z = 0; Real sq_cum_grad = 0; };

struct FTRLHandle : public ISGDHandle {
 public:
  inline void Init(FeaID key,  FTRLEntry& val) { }

  inline void Push(FeaID key, Blob<const Real> grad, FTRLEntry& val) {
    // update cum grad
    Real g = grad[0];
    Real sqrt_n = val.sq_cum_grad;
    val.sq_cum_grad = sqrt(sqrt_n * sqrt_n + g * g);

    // update z
    Real old_w = val.w;
    Real sigma = (val.sq_cum_grad - sqrt_n) / this->alpha;
    val.z += g - sigma * old_w;

    // update w
    val.w = this->penalty.Solve(-val.z, (this->beta + val.sq_cum_grad) / this->alpha);

    // update monitor
    this->tracker->Update(val.w, old_w);
  }

  inline void Pull(FeaID key, const FTRLEntry& val, Blob<Real>& send) {
    send[0] = val.w;
  }
};
}  // namespace linear
}  // namespace dmlc

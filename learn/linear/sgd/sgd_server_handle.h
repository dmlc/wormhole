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

/**
 * \brief the base handle class
 */
template <typename K, typename V>
struct ISGDHandle {
 public:
  inline void Pull(ps::Blob<const K> recv_key,
                   ps::Blob<const V> my_val,
                   ps::Blob<V> send_val) {
    send_val[0] = my_val[0];
  }

  inline void Start(bool push, int timestamp, void* msg) { }
  inline void Finish() { tracker->Report(); }
  inline void SetCaller(void *obj) { }

  ModelMonitor* tracker = nullptr;
  L1L2<V> penalty;

  // learning rate
  V alpha = 0.1, beta = 1;
};

/**
 * \brief The standard SGD handle.
 * use alpha / ( beta + sqrt(t)) as the learning rate
 */
template <typename K, typename V>
struct SGDHandle : public ISGDHandle<K,V> {
 public:
  template <typename T> using Blob = ps::Blob<T>;

  inline void Init(Blob<const K> key, Blob<V> val) {
    val[0] = 0;
  }

  inline void Start(bool push, int timestamp, void* msg) {
    if (push) {
      eta = (this->beta + sqrt(t)) / this->alpha;
      t += 1;
    }
  }

  inline void Push(
      Blob<const K> recv_key, Blob<const V> recv_val, Blob<V> my_val) {
    my_val[0] = this->penalty.Solve(eta * my_val[0] - recv_val[0], eta);
  }

  // iteration count
  V t = 1;
  V eta = 0;
};


/**
 * \brief AdaGrad SGD handle.
 * use alpha / ( beta + sqrt(sum_t grad_t^2)) as the learning rate
 *
 * my_val is a length-2 vector,
 * val[0]: weight
 * val[1]: sqrt(sum_t grad_t^2)
 */
template <typename K, typename V>
struct AdaGradHandle : public ISGDHandle<K, V>{
  template <typename T> using Blob = ps::Blob<T>;

  inline void Init(Blob<const K> key, Blob<V> val) {
    val[0] = 0;
    val[1] = 0;
  }

  inline void Push(
      Blob<const K> recv_key, Blob<const V> recv_val, Blob<V> my_val) {
    V grad = recv_val[0];
    V* val = my_val.data;
    V sqrt_n = val[1];
    val[1] = sqrt(sqrt_n * sqrt_n + grad * grad);
    V eta = (val[1] + this->beta) / this->alpha;
    V w = my_val[0];
    my_val[0] = this->penalty.Solve(eta * w - grad, eta);
    this->tracker->Update(my_val[0], w);
  }
};


/**
 * \brief FTRL updater, use a smoothed weight for better spasity
 *
 * my_val is a length-3 vector,
 * val[0]: weight
 * val[1]: z, the smoothed version of - eta * w + grad
 * val[2]: sqrt(sum_t grad_t^2)
 */
template <typename K, typename V>
struct FTRLHandle : public ISGDHandle<K, V> {
 public:
  template <typename T> using Blob = ps::Blob<T>;

  inline void Init(Blob<const K> key, Blob<V> val) {
    val[0] = 0;
    val[1] = 0;
    val[2] = 0;
  }

  inline void Push(
      Blob<const K> recv_key, Blob<const V> recv_val, Blob<V> my_val) {
    V* val = my_val.data;

    // update cum grad
    V grad = recv_val[0];
    V sqrt_n = val[2];
    val[2] = sqrt(sqrt_n * sqrt_n + grad * grad);

    // update z
    V w = val[0];
    V sigma = (val[2] - sqrt_n) / this->alpha;
    val[1] += grad  - sigma * w;

    // update w
    val[0] = this->penalty.Solve(-val[1], (this->beta + val[2]) / this->alpha);

    // update monitor
    this->tracker->Update(val[0], w);
  }
};
}  // namespace linear
}  // namespace dmlc

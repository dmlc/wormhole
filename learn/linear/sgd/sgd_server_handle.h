/**
 * @file   sgd_server_handle.h
 * @brief  server handles for sgd.
 *
 * No class inheritance and virtual function for better performance
 */
#include "base/monitor.h"
#include "base/penalty.h"
#include "ps/blob.h"
namespace dmlc {
namespace linear {

/**
 * \brief The basic SGD handle.
 * use alpha / ( beta + sqrt(t)) as the learning rate
 */
template <typename K, typename V>
struct SGDHandle {
 public:
  template <typename T> using Blob = ps::Blob<T>;

  inline void Init(Blob<const K> key, Blob<V> val) {
    val[0] = 0;
  }

  inline void Start(bool push, int timestamp, void* msg) {
    if (push) {
      t += 1;
      eta = (beta + sqrt(t)) / alpha;
    }
  }

  inline void Push(
      Blob<const K> recv_key, Blob<const V> recv_val, Blob<V> my_val) {
    my_val[0] = penalty.Solve(eta * my_val[0] - recv_val[0], eta);
  }

  inline void Pull(
      Blob<const K> recv_key, Blob<const V> my_val, Blob<V> send_val) {
    send_val[0] = my_val[0];
  }


  inline void Finish() { tracker->Report(); }

  ModelMonitor* tracker = nullptr;
  L1L2<V> penalty;
  // learning rate
  V alpha = 0.1, beta = 1;
  // iteration count
  V t = 0;
  V eta = 0;

  // empty funcs
  inline void SetCaller(void *obj) { }
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
struct AdaGradHandle {
  template <typename T> using Blob = ps::Blob<T>;

  inline void Init(Blob<const K> key, Blob<V> val) {
    val[0] = 0;
    val[1] = 0;
  }

  inline void Push(
      Blob<const K> recv_key, Blob<const V> recv_val, Blob<V> my_val) {
    V grad = recv_val[0];
    V* val = my_val.data;
    V sqrt_n = val[2];
    val[2] = sqrt(sqrt_n * sqrt_n + grad * grad);
    V eta = (val[2] + beta) / alpha;
    my_val[0] = penalty.Solve(eta * my_val[0] - grad, eta);
  }

  inline void Pull(
      Blob<const K> recv_key, Blob<const V> my_val, Blob<V> send_val) {
    send_val[0] = my_val[0];
  }

  inline void Finish() { tracker->Report(); }

  ModelMonitor* tracker = nullptr;
  L1L2<V> penalty;
  // learning rate
  V alpha = 0.1, beta = 1;

  // empty funcs
  inline void Start(bool push, int timestamp, void* msg) { }
  inline void SetCaller(void *obj) { }
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
struct FTRLHandle {
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
    V sigma = (val[2] - sqrt_n) / alpha;
    val[1] += grad  - sigma * w;

    // update w
    val[0] = penalty.Solve(-val[1], (beta + val[2]) / alpha);

    // update monitor
    DCHECK(tracker);
    tracker->Update(val[0], w);
  }

  inline void Pull(
      Blob<const K> recv_key, Blob<const V> my_val, Blob<V> send_val) {
    send_val[0] = my_val[0];
  }

  inline void Finish() {
    // LOG(ERROR) << tracker->prog.PrintStr();
    tracker->Report();
  }


  // learning rate
  V alpha = 0.1, beta = 1;

  L1L2<V> penalty;

  ModelMonitor* tracker = nullptr;

  // empty funcs
  inline void SetCaller(void *obj) { }
  inline void Start(bool push, int timestamp, void* msg) { }

};
}  // namespace linear
}  // namespace dmlc

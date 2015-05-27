/**
 * @file   delay_tol_handle.h
 * @brief  experimental delay tolerante handles.
 */
#pragma once
#include "base/monitor.h"
#include "base/penalty.h"
#include "ps/blob.h"
#include "system/message.h"
#include "filter/filter.h"
#include "sgd/sgd_server_handle.h"
#include <city.h>
namespace dmlc {
namespace linear {

using ps::Message;

// find the signature of the keys list
inline uint64_t Signature(ps::Message* msg) {
  ps::Filter* f = ps::IFilter::Find(ps::Filter::KEY_CACHING, msg);
  if (f) {
    CHECK(f->has_signature());
    return f->signature();
  } else {
    return CityHash64(msg->key.data(), msg->key.size());
  }
}

/// Version 1, use grad^bak

/// Version 2, use sqrt(t + tau(t)) in learning rate

// get and set tau(t)
class TimeDelay {
 public:
  void Store(int t, ps::Message* msg) {
    auto sig = std::make_pair(msg->sender, Signature(msg));
    vc_[sig] = t;
  }
  int Fetch(ps::Message* msg) {
    auto sig = std::make_pair(msg->sender, Signature(msg));
    auto it = vc_.find(sig);
    CHECK(it != vc_.end()) << sig.first << ", " << sig.second;
    // vc_.erase(it);
    return it->second;
  }
 private:
  // store timestamp
  std::map<std::pair<std::string, uint64_t>, int> vc_;
};

/**
 * \brief The basic SGD handle.
 * use alpha / ( beta + sqrt(t)) as the learning rate
 */
template <typename K, typename V>
struct DTSGDHandle : public SGDHandle<K,V> {
 public:
  template <typename T> using Blob = ps::Blob<T>;

  inline void Start(bool push, int timestamp, void* msg) {
    Message* m = (Message*) msg;
    if (push) {
      int tau = std::max(this->t - delay.Fetch(m), 0);
      LOG(INFO) << m->sender << " " << this->t << " " << tau;
      this->eta = (this->beta + sqrt(this->t + tau)) / this->alpha;
      this->t += 1;
    } else {
      delay.Store(this->t, m);
    }
  }
  TimeDelay delay;
};


template <typename K, typename V>
struct DTAdaGradHandle : public AdaGradHandle<K, V> {

  inline void Start(bool push, int timestamp, void* msg) {
    Message* m = (Message*) msg;
    if (push) {
      int tau = std::max(t - delay.Fetch(m), 0);
      LOG(INFO) << m->sender << " " << t << " " << tau;
      adjust = sqrt(t + tau) / sqrt((V)t);
      t += 1;
    } else {
      delay.Store(t, m);
    }
  }

  inline void Push(
      ps::Blob<const K> recv_key, ps::Blob<const V> recv_val, ps::Blob<V> my_val) {
    V grad = recv_val[0];
    V* val = my_val.data;
    V sqrt_n = val[1];
    val[1] = sqrt(sqrt_n * sqrt_n + grad * grad);
    V eta = (val[1] * adjust + this->beta) / this->alpha;
    V w = my_val[0];
    my_val[0] = this->penalty.Solve(eta * w - grad, eta);
    this->tracker->Update(my_val[0], w);
  }

  int t = 1;
  V adjust;
  TimeDelay delay;
};

}  // namespace linear
}  // namespace dmlc

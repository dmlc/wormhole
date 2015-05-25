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
namespace dmlc {
namespace linear {

using ps::Message;

// find the signature of the keys list
inline uint64_t Signature(ps::Message* msg) {
  ps::Filter* f = CHECK_NOTNULL(
      ps::IFilter::Find(ps::Filter::KEY_CACHING, msg));
  CHECK(f->has_signature());
  return f->signature();
}

/// Version 1, use sqrt(t + tau(t)) in learning rate
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
    auto sig = std::make_pair(m->sender, Signature(m));
    if (push) {
      auto it = vc_.find(sig);
      CHECK(it != vc_.end()) << sig.first << ", " << sig.second;
      V tau = this->t - it->second;
      LOG(INFO) << sig.first << " " << this->t << " " << tau;
      this->eta = (this->beta + sqrt(this->t + tau)) / this->alpha;
      this->t += 1;
    } else {
      // store the timestamp
      vc_[sig] = this->t;
    }
  }

  // store timestamp
  std::map<std::pair<std::string, uint64_t >, V> vc_;

};
}  // namespace linear
}  // namespace dmlc

/**
 * @file   sgd_server_handle.h
 * @brief  server handles for sgd
 */
#include "base/monitor.h"
#include "../../repo/ps-lite/src/base/blob.h"
namespace dmlc {
namespace linear {

// /**
//  * @brief Track the progress
//  */
// struct OnlineModelTracker {
//  public:
//   inline void Send() {
//   //   if (!reporter) return;
//   //   reporter->Report(prog);
//   //   prog.fvec[3] = 0;
//   //   prog.fvec[4] = 0;
//   }
//   // ivec[1] : nnz(w), fvec[3] : |w|^2_2, fvec[4] : |delta_w|^2_2
//   template<typename V>
//   inline void Update(V cur, V old) {
//     if (cur == 0) {
//       if (old == 0) {
//         return;
//       } else {
//         -- prog.ivec[1];
//         prog.fvec[4] += old * old;
//       }
//     } else {
//       V cc = cur * cur;
//       prog.fvec[3] += cc;
//       if (old == 0) {
//         ++ prog.ivec[1];
//         prog.fvec[4] += cc;
//       } else {
//         V delta = cur - old;
//         prog.fvec[4] += delta * delta;
//       }
//     }
//   }

//   Progress prog;
//   // MonitorSlaver<Progress>* reporter = nullptr;
// };

/**
 * \brief FTRL updater
 *
 * my_val is a length-3 vector, 0: weight, 1: z, 2: square rooted cumulatived
 * gradient
 */

template <typename K, typename V>
struct FTRLHandle {
 public:
  template <typename T> using Blob = ps::Blob<T>;

  inline void HandlePush(
      int ts, Blob<const K> recv_key, Blob<const V> recv_val, Blob<V> my_val) {
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
    V z = val[1];
    if (z <= lambda1 && z >= -lambda1) {
      val[0] = 0;
    } else {
      val[0] = - (z - (z > 0 ? 1 : -1) * lambda1) /
               ((beta + val[2]) / alpha + lambda2);
    }

    // update monitor
    DCHECK(tracker);
    tracker->Update(val[0], w);
  }

  inline void HandlePull(
      int ts, Blob<const K> recv_key, Blob<const V> my_val, Blob<V> send_val) {
    send_val[0] += my_val[0];
  }

  inline void HandleInit(int ts, Blob<const K> key, Blob<V> val) {
    val[0] = 0;
    val[1] = 0;
    val[2] = 0;
  }

  // learning rate
  V alpha = 0.1, beta = 1;

  // penalty, lambda1 * |w|_1 + lambda2 * ||w||_2^2
  V lambda1 = 1, lambda2 = .1;

  ModelMonitor* tracker = nullptr;
};
}  // namespace linear
}  // namespace dmlc

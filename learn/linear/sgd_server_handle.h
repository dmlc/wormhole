/**
 * @file   sgd_server_handle.h
 * @brief  server handles for sgd
 */
namespace dmlc {
namespace linear {

/**
 * @brief Track the progress
 */
struct OnlineModelTracker {
 public:
  inline void Send() {
    if (!reporter) return;
    reporter->Report(prog);
    prog.fdata[3] = 0;
    prog.fdata[4] = 0;
  }

  inline template<typename V>
  void Update(V cur, V old) {
    if (cur == 0 && old != 0) {
      -- prog.idata[1];
    } else if (cur != 0 && old == 0) {
      ++ prog.idata[1];
    }
    prog.fdata[3] += cur * cur;
    V delta = cur - old;
    prog.fdata[4] += delta * delta;
  }

  Progress prog;
  MonitorSlaver<Progress>* reporter = nullptr;
};

/**
 * \brief FTRL updater
 *
 * my_val is a length-3 vector, 0: weight, 1: z, 2: square rooted cumulatived
 * gradient
 */
template <typename V>
struct FTRLHandle {
 public:
  inline void HandlePush(
      int ts, CBlob<Key> recv_keys, CBlob<V> recv_vals, Blob<V>* my_vals) {
    DCHECK_EQ(my_vals->size, 3);
    DCHECK_EQ(recv_vals.size, 1);
    V* val = my_vals->data;

    // update cum grad
    V grad = recv_vals.data[0];
    V sqrt_n = val[2];
    val[2] = sqrt(sqrt_n * sqrt_n + grad * grad);

    // update z
    V w = val[0];
    V sigma = (val[2] - sqrt_n) / alpha;
    val[1] += grad  - sigma * w;

    // update w
    V eta = alpha / (val[2] + beta);
    V t = lambda1 * eta;
    V u = - val[1] * eta;
    if (u <= t && u >= -t) {
      val[0] = 0;
    } else if (u > 0) {
      val[0] = (u + (u > 0 ? -t : t)) / ( 1 + lambda2 * eta);
    }

    // update monitor
    DCHECK(tracker);
    tracker->Update(val[0], w);
  }

  inline void HandlePull(
      int ts, CBlob<Key> recv_keys, CBlob<V> my_vals, Blob<V>* send_vals) {
    DCHECK_EQ(my_vals.size, 3);
    DCHECK_EQ(send_vals->size, 1);
    send_vals->data[0] += my_vals.data[0];
  }

  inline void HandleInit(int ts, CBlob<Key> keys, Blob<V>* vals) {
    DCHECK_EQ(vals->size, 3);
    vals->data[0] = 0;
    vals->data[1] = 0;
    vals->data[2] = 0;
  }

  // learning rate
  V alpha = 0.1, beta = 1;

  // penalty, lambda1 * |w|_1 + lambda2 * ||w||_2^2
  V lambda1 = 1, lambda2 = .1;

  OnlineModelTracker* tracker = nullptr;
};
}  // namespace linear
}  // namespace dmlc

#pragma once
#include <cstring>
#include <sstream>
#include "dmlc/logging.h"
namespace dmlc {


/**
 * @brief Monitor the model
 *
 * ivec[1] : nnz(w), fvec[3] : |w|^2_2, fvec[4] : |delta_w|^2_2
 */
struct ModelMonitor {
  virtual ~ModelMonitor() { }

  template<typename V>
  inline void Update(V cur_w, V old_w) {
    if (cur_w == 0) {
      if (old_w == 0) {
        return;
      } else {
        -- prog.nnz_w();
        prog.wdelta2() += old_w * old_w;
      }
    } else {
      V cc = cur_w * cur_w;
      prog.weight2() += cc;
      if (old_w == 0) {
        ++ prog.nnz_w();
        prog.wdelta2() += cc;
      } else {
        V delta = cur_w - old_w;
        prog.wdelta2() += delta * delta;
      }
    }
  }

  template<typename V>
  inline void Update(const std::vector<V>& cur_w, const std::vector<V>& old_w) {
    CHEK_EQ(cur_w.size(), old_w.size());
    for (size_t i = 0; i < cur_w.size(); ++i) {
      Update(cur_w[i], old_w[i]);
    }
  }
  virtual void Report() { }

  void Clear() { prog.Clear(); }
  Progress prog;
};

/**
 * @brief Monitor a worker
 *
 * ivec[0 : #example, fvec[0]: objv, fvec[1]: rel_objv |w|^2_2, fvec[4] : |delta_w|^2_2
 */

struct WorkerMonitor {
  template <typename V>
  void Update(size_t num_ex, ScalarLoss<V>* loss) {
    prog.num_ex() += num_ex;
    prog.objv() += loss->Objv();
    prog.acc() += loss->Accuracy();
    prog.auc() += loss->AUC();
    prog.count() ++;
  }

  void Clear() { prog.Clear(); }
  Progress prog;
};

}  // namespace dmlc

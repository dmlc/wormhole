#pragma once
#include <cstring>
#include <sstream>
#include "dmlc/logging.h"
namespace dmlc {
namespace linear {

/**
 * @brief objective values, running time, ...
 */
class Progress {
 public:
  Progress() {
    fvec.resize(kfnum);
    fsize = kfnum * sizeof(double);
    ivec.resize(kinum);
    isize = kinum * sizeof(size_t);
  }

  void Clear() {
    std::memset(fvec.data(), 0, fsize);
    std::memset(ivec.data(), 0, isize);
  }

  void Merge(const Progress& other) {
    for (int i = 0; i < kfnum; ++i) {
      fvec[i] += other.fvec[i];
    }
    for (int i = 0; i < kfnum; ++i) {
      ivec[i] += other.ivec[i];
    }
  }

  void Parse(const std::string& str) {
    CHECK_EQ(str.size(), fsize + isize);
    std::memcpy(fvec.data(), str.data(), fsize);
    std::memcpy(ivec.data(), str.data()+fsize, isize);
  }

  void Serialize(std::string* str) const {
    str->clear();
    str->append((char*)fvec.data(), fsize);
    str->append((char*)ivec.data(), isize);
  }

  std::string PrintStr() {
    std::stringstream ss;
    if (num_ex() == 0) {
      ss << " no update ";
    } else {
      ss << "objv " << objv() / num_ex()
         << ", auc " << auc() / count()
         << ", acc " << acc() / count()
         << ", nnz w " << nnz_w();
    }
    return ss.str();
  }

  // mutator
  double& objv() { return fvec[0]; }
  double& acc() { return fvec[1]; }
  double& auc() { return fvec[2]; }
  double& weight2() { return fvec[3]; }
  double& wdelta2() { return fvec[4]; }

  size_t& count() { return ivec[0]; }
  size_t& num_ex() { return ivec[1]; }
  size_t& nnz_w() { return ivec[2]; }


 private:
  static const int kfnum = 10;
  static const int kinum = 10;

  std::vector<double> fvec;
  std::vector<size_t> ivec;

  size_t fsize;
  size_t isize;
};

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


}  // namespace linear
}  // namespace dmlc

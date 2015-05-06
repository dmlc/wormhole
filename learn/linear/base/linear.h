#include <cstring>
namespace dmlc {
namespace linear {

// #include "../../repo/ps-lite/src/base/blob.h"
// using ps::Key FeaID;

// commands
static const int kRequestWorkload = 1;


/**
 * @brief objective values, running time, ...
 */
struct Progress {
  static const int kfnum = 10;
  static const int kinum = 10;
  Progress() {
    fvec.resize(kfnum);
    ivec.resize(kinum);
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

  void Serialize(std::string* str) {
    str->clear();
    str->append((char*)fvec.data(), fsize);
    str->append((char*)ivec.data(), isize);
  }

  std::vector<double> fvec;
  std::vector<size_t> ivec;

  size_t fsize;
  size_t isize;
};
}  // namespace linear
}  // namespace dmlc

namespace dmlc {
namespace linear {

// commands
static const int kRequestWorkload = 1;


/**
 * @brief objective values, running time, ...
 */
struct Progress {
  static const int kfnum = 10;
  static const int kinum = 10;
  Progress() {
    fsize = kfnum*sizeof(double);
    isize = kinum*sizeof(uint64);
  }

  void Clear() {
    memset(fdata, 0, fsize);
    memset(idata, 0, isize);
  }

  void Merge(const Process& other) {
    for (int i = 0; i < kfnum; ++i) {
      fdata[i] += other.fdata[i];
    }
    for (int i = 0; i < finum; ++i) {
      idata[i] += other.idata[i];
    }
  }

  void Parse(const std::string& str) {
    CHECK_EQ(str.size(), fsize + isize);
    memcpy(fdata, str.data(), fsize);
    memcpy(idata, str.data()+fsize, isize);
  }

  void Serialize(std::string* str) {
    str->resize(fsize, isize);
    memcpy(str->data(), fdata, fsize);
    memcpy(str->data()+fdata, idata, isize);
  }

  double fdata[kfnum];
  uint64 idata[kinum];

  size_t fsize;
  size_t isize;
};
}  // namespace linear
}  // namespace dmlc

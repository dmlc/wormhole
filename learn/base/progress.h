/**
 * \file   progress.h
 * \brief The algorithm progress a worker/server report to the scheduler. It
 * should be printable, serializable, and mergable.
 */
#pragma once
#include <string>
#include <vector>
#include "dmlc/logging.h"
namespace dmlc {

/**
 * \brief  The interface of progress
 */
class IProgress {
 public:
  IProgress() { }
  virtual ~IProgress() { }

  virtual void Clear() = 0;
  virtual bool Empty() const = 0;

  /// merger from another progress
  virtual void Merge(const Progress* const other) = 0;

  /// parse from string
  virtual void Parse(const std::string& str) = 0;

  /// serialize to string
  virtual void Serialize(std::string* str) = 0;

  /// head string for printing
  virtual std::string HeadStr() = 0;

  /// string for printing
  virtual std::string PrintStr(const Progress* const prev) = 0;
};


class VectorProgress : public IProgress {
 public:
  VectorProgress() { Resize(0, 0); }
  VectorProgress(int inum, int fnum) { Resize(inum, fnum); }
  virtual ~VectorProgress() { }

  void Resize(int inum, int fnum) {
    fvec_.resize(fnum);
    ivec_.resize(inum);
    fsize_ = fnum * sizeof(double);
    isize_ = inum * sizeof(size_t);
  }

  virtual bool Empty() const {
    for (double f : fvec_) if (f != 0) return false;
    for (size_t i : ivec_) if (i != 0) return false;
    return true;
  }

  virtual void Clear() {
    std::memset(fvec_.data(), 0, fsize_);
    std::memset(ivec_.data(), 0, isize_);
  }

  virtual void Parse(const std::string& str) {
    size_t fs, is;
    char const* ptr = str.data();
    CHECK_GE(str.size(), sizeof(size_t)*2);
    std::memcpy(&is, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    std::memcpy(&fs, ptr, sizeof(size_t));
    ptr += sizeof(size_t);
    CHECK_EQ(fs, fsize_);
    CHECK_EQ(is, isize_);
    CHECK_EQ(str.size(), fs + is + sizeof(size_t)*2);
    std::memcpy(ivec_.data(), ptr, isize_);
    ptr += isize_;
    std::memcpy(fvec_.data(), ptr, fsize_);
  }

  virtual void Serialize(std::string* str) const {
    str->clear();
    str->append((char*)&isize_, sizeof(size_t));
    str->append((char*)&fsize_, sizeof(size_t));
    str->append((char*)ivec_.data(), isize_);
    str->append((char*)fvec_.data(), fsize_);
  }

 protected:
  std::vector<double> fvec_;
  std::vector<size_t> ivec_;

 private:
  size_t fsize_;
  size_t isize_;
};

}  // namespace dmlc

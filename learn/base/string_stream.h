/**
 * @file   string_stream.h
 * @brief
 */
#pragma once
#include "dmlc/io.h"
#include "dmlc/logging.h"
namespace dmlc {

class StringStream : public Stream {
 public:
  StringStream() : pos_(0) { }

  explicit StringStream(const std::string& str) : buf_(str), pos_(0) { }

  virtual ~StringStream() { }

  virtual size_t Read(void *ptr, size_t size) {
    CHECK_LE(size+pos_, buf_.size());
    memcpy(ptr, buf_.data() + pos_, size);
    pos_ += size;
    return size;
  }

  virtual void Write(const void *ptr, size_t size) {
    buf_.append((const char*)ptr, size);
  }

  std::string str() { return buf_; }
 private:
  std::string buf_;
  size_t pos_;
};

}  // namespace dmlc

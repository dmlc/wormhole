/**
 * \file   compressed_row_block.h
 * \brief  read/write compressed row block from/to recordio files
 */
#pragma once
#include "lz4.h"
#include "data/row_block.h"

namespace dmlc {
namespace data {

/**
 * \brief compress and decompress a row block
 */
class CompressedRowBlock {
 public:
  template <typename IndexType>
  void Compress(RowBlock<IndexType> blk, std::string* str) {
    str->clear();
    str->reserve(MaxCompressionSize(blk));
    str_ = str;
    // data_ = str->data(); cur_len_ = 0; max_len_ = str->size();

    int nrows = blk.size;
    int nnz = blk.offset[nrows] - blk.offset[0];

    Write(kMagicNumber);
    Write(sizeof(IndexType));
    Write(nrows);
    Compress((const char*)blk.label, nrows * sizeof(real_t));
    Compress((const char*)blk.offset, (nrows+1) * sizeof(size_t));
    Compress((const char*)blk.index, nnz * sizeof(IndexType));
    Compress((const char*)blk.value, nnz * sizeof(real_t));
    Compress((const char*)blk.weight, nrows * sizeof(real_t));
  }

  template <typename IndexType>
  void Decompress(const std::string&str,
                  RowBlockContainer<IndexType>* blk) {
    Decompress(str.data(), str.size(), blk);
  }

  template <typename IndexType>
  void Decompress(char const* data, size_t size,
                  RowBlockContainer<IndexType>* blk) {
    cdata_ = data; cur_len_ = 0; max_len_ = size;
    CHECK_EQ(Read(), kMagicNumber) << "wrong data format";
    CHECK_EQ(Read(), (int)sizeof(IndexType)) << "wrong indextype";

    int nrows = Read();
    Decompress(&blk->label, nrows);
    Decompress(&blk->offset, nrows + 1);

    CHECK_EQ(blk->offset.size(), nrows+1);
    int nnz = blk->offset[nrows] - blk->offset[0];
    Decompress(&blk->index, nnz);
    Decompress(&blk->value, nnz);
    Decompress(&blk->weight, nrows);
  }

 private:
  template <typename IndexType>
  size_t MaxCompressionSize(RowBlock<IndexType> blk) {
    int nrows = blk.size;
    int nnz = blk.offset[nrows] - blk.offset[0];
    size_t size = 6 * sizeof(int)  // size
                  + LZ4_compressBound((nrows+1)*sizeof(size_t));  // offset
    if (blk.label) size += LZ4_compressBound(nrows*sizeof(size_t));
    if (blk.index) size += LZ4_compressBound(nnz*sizeof(IndexType));
    if (blk.value) size += LZ4_compressBound(nnz*sizeof(real_t));
    if (blk.weight) size += LZ4_compressBound(nrows*sizeof(real_t));
    return size;
  }

  void Compress(const char* data, size_t size) {
    if (data == NULL) { Write(0); return; }

    int dst_size = LZ4_compressBound(size);
    char *dst = new char[dst_size];
    int actual_size = LZ4_compress_default(data, dst, size, dst_size);
    CHECK_NE(actual_size, 0);

    Write(actual_size);
    str_->append(dst, actual_size);
  }

  template <typename T>
  void Decompress(std::vector<T>* dst, int len) {
    int cp_size = Read();
    if (cp_size <= 0) return;
    CHECK_LE(cur_len_ + cp_size, max_len_);
    dst->resize(len);
    int dst_size = len * sizeof(T);
    CHECK_EQ(dst_size, LZ4_decompress_safe(
        cdata_ + cur_len_, dst->data(), cp_size, dst_size));
    cur_len_ += cp_size;
  }

  void Write(int num) {
    str_->append((const char*)&num, sizeof(int));
  }

  int Read() {
    CHECK_LE(cur_len_ + sizeof(int), max_len_);
    int ret;
    memcpy(&ret, cdata_+cur_len_, sizeof(int));
    cur_len_ += sizeof(int);
    return ret;
  }

  // char* data_;
  std::string* str_;
  char const* cdata_;
  size_t max_len_, cur_len_;
  static const int kMagicNumber = 1196140743;
};


} // namespace data
} // namespace dmlc

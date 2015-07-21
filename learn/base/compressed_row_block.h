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
  void Compress(RowBlock<IndexType> blk, std::string* str) {
    str->resize(MaxCompressionSize());
    data_ = str->data(); cur_len_ = 0; max_len_ = str->size();

    int nrows = blk.size;
    int nnz = blk.offset[nrow] - blk.offset[0];

    Write(nrows);
    Compress(blk.label, nrows * sizeof(real_t));
    Compress(blk.offset, (nrows+1) * sizeof(size_t));
    Compress(blk.index, nnz * sizeof(IndexType));
    Compress(blk.value, nnz * sizeof(real_t));
    Compress(blk.weight, nrows * sizeof(real_t));

    str->resize(cur_len_);
  }

  void Decompress(const std::string&str,
                  RowBlockContainer<IndexType>* blk) {
    Decompress(str.data(), str.size(), blk);
  }

  void Decompress(const char const* data, size_t size,
                  RowBlockContainer<IndexType>* blk) {
    cdata_ = data; cur_len_ = 0; max_len_ = size;
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
  size_t MaxCompressionSize(RowBlock<IndexType> blk) {
    int nrows = blk.size;
    int nnz = blk.offset[nrow] - blk.offset[0];
    size_t size = 6 * sizeof(int)  // size
                  + LZ4_compressBound((nrows+1)*sizeof(size_t));  // offset
    if (blk.label) size += LZ4_compressBound(nrows*sizeof(size_t));
    if (blk.index) size += LZ4_compressBound(nnz*sizeof(IndexType));
    if (blk.value) size += LZ4_compressBound(nnz*sizeof(real_t));
    if (blk.weight) size += LZ4_compressBound(nrows*sizeof(real_t));
    return size;
  }

  void Compress(const char* data, char* size) {
    int dst_size = LZ4_compressBound(size);
    char *dst = new char[dst_size];
    int actual_size = LZ4_compress_default(data, dst, size, dst_size);
    CHECK_NE(actual_size, 0);

    Write(actual_size);
    CHECK_LE(actual_size + cur_len_, max_len_);
    memcpy(data_ + cur_len_, dst, actual_size);
    cur_len_ + actual_size;
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
    CHECK_LE(cur_len_ + sizeof(int), max_len_);
    memcpy(data_+cur_len_, &num, sizeof(int));
    cur_len_ += sizeof(int);
  }

  int Read() {
    CHECK_LE(cur_len_ + sizeof(int), max_len_);
    int ret;
    memcpy(&ret, cdata_+cur_len_, sizeof(int));
    cur_len_ += sizeof(int);
  }

  char* data_;
  char const* cdata_;
  size_t max_len_, cur_len_;
};


} // namespace data
} // namespace dmlc

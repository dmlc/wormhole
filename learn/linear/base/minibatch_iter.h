/**
 * @file   minibatch_iter.h
 * @brief  A minibatch iterator
 */
#pragma once
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <cstring>
#include "data/row_block.h"
#include "data/parser.h"
#include "data/libsvm_parser.h"
#include "base/utils.h"
namespace dmlc {
namespace data {

/**
 * \brief Read a fixed size minibatch each time.
 *
 * the current implementation is not efficient due to unnecessary data copy
 */
template<typename IndexType>
class MinibatchIter {
 public:
  MinibatchIter(const char* uri, unsigned part_index, unsigned num_parts,
                const char* type, unsigned minibatch_size)
      : mb_size_(minibatch_size), start_(0), end_(0) {
    // create parser
    if (!strcmp(type, "libsvm")) {
      parser_ = new LibSVMParser<IndexType>(InputSplit::Create(
          uri, part_index, num_parts, "text"), 1);
    } else {
      LOG(FATAL) << "unknown datatype " << type;
    }
	parser_ = new ThreadedParser<IndexType>(parser_);
  }

  virtual ~MinibatchIter() {
    delete parser_;
  }

  void BeforeFirst(void) {
    parser_->BeforeFirst();
  }

  bool Next(void) {
    mb_.Clear();
    while (mb_.offset.size() < mb_size_ + 1) {
      if (start_ == end_) {
        if (!parser_->Next()) break;
        in_blk_ = parser_->Value();
        start_ = 0;
        end_ = in_blk_.size;
      }
      size_t len = std::min(end_ - start_, mb_size_ + 1 - mb_.offset.size());
      Push(start_, len);
      start_ += len;
    }
    out_blk_ = mb_.GetBlock();
    return out_blk_.size > 0;
  }

  const RowBlock<IndexType> &Value(void) const {
    return out_blk_;
  }

 private:
  void Push(size_t pos, size_t len) {
    if (!len) return;
    CHECK_LE(pos + len, in_blk_.size);
    RowBlock<IndexType> slice;
    slice.size = len;
    slice.offset  = in_blk_.offset + pos;
    slice.label   = in_blk_.label  + pos;
    slice.index   = in_blk_.index  + in_blk_.offset[pos];
    if (in_blk_.value)
      slice.value = in_blk_.value  + in_blk_.offset[pos];

    // LOG(INFO) << DebugStr(slice);
    mb_.Push(slice);
  }

  unsigned mb_size_;
  Parser<IndexType> *parser_;

  size_t start_, end_;
  RowBlock<IndexType> in_blk_;
  RowBlockContainer<IndexType> mb_;
  RowBlock<IndexType> out_blk_;
};

}  // namespace data
}  // namespace dmlc

/**
 * @file   minibatch_iter.h
 * @brief  A minibatch iterator
 */
#pragma once
#include <algorithm>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <cstring>
#include "data/row_block.h"
#include "data/parser.h"
#include "data/libsvm_parser.h"
#include "base/adfea_parser.h"
#include "base/criteo_parser.h"
#include "base/crb_parser.h"
#include "base/debug.h"
namespace dmlc {
namespace data {

/**
 * \brief Read a fixed size minibatch each time.
 * @param minibatch_size the minibatch size
 * @param if nonzero, then the minibatch is randomly picked from a buffer with
 * *shuf_buf* examples
 */
template<typename IndexType>
class MinibatchIter {
 public:
  MinibatchIter(const char* uri, unsigned part_index, unsigned num_parts,
                const char* type, unsigned minibatch_size, unsigned shuf_buf = 0)
      : mb_size_(minibatch_size), shuf_buf_(shuf_buf), start_(0), end_(0) {
    if (shuf_buf) {
      CHECK_GT(shuf_buf, minibatch_size);
      buf_reader_ =
          new MinibatchIter(uri, part_index, num_parts, type, shuf_buf, 0);
      parser_ = NULL;
    } else {
      // create parser
      if (!strcmp(type, "libsvm")) {
        parser_ = new LibSVMParser<IndexType>(
            InputSplit::Create(uri, part_index, num_parts, "text"), 1);
      } else if (!strcmp(type, "criteo")) {
        parser_ = new CriteoParser<IndexType>(
            InputSplit::Create(uri, part_index, num_parts, "text"), true);
      } else if (!strcmp(type, "criteo_test")) {
        parser_ = new CriteoParser<IndexType>(
            InputSplit::Create(uri, part_index, num_parts, "text"), false);
      } else if (!strcmp(type, "adfea")) {
        parser_ = new AdfeaParser<IndexType>(
            InputSplit::Create(uri, part_index, num_parts, "text"));
      } else if (!strcmp(type, "crb")) {
        parser_ = new CRBParser<IndexType>(
            InputSplit::Create(uri, part_index, num_parts, "recordio"));
      } else {
        LOG(FATAL) << "unknown datatype " << type;
      }
      parser_ = new ThreadedParser<IndexType>(parser_);
      buf_reader_ = NULL;
    }
  }

  virtual ~MinibatchIter() {
    delete parser_;
    delete buf_reader_;
  }

  void BeforeFirst(void) {
    if (parser_) parser_->BeforeFirst();
    if (buf_reader_) buf_reader_->BeforeFirst();
  }

  bool Next(void) {
    mb_.Clear();
    while (mb_.offset.size() < mb_size_ + 1) {
      if (start_ == end_) {
        if (shuf_buf_ == 0) {
          // no random shuffle
          if (!parser_->Next()) break;
          in_blk_ = parser_->Value();

        } else {
          // do random shuffle
          if (!buf_reader_->Next()) break;
          in_blk_ = buf_reader_->Value();
          if (rdp_.size() != in_blk_.size) {
            rdp_.resize(in_blk_.size);
            for (size_t i = 0; i < in_blk_.size; ++i) rdp_[i] = i;
          }
          std::random_shuffle(rdp_.begin(), rdp_.end());
        }
        start_ = 0;
        end_ = in_blk_.size;
      }

      size_t len = std::min(end_ - start_, mb_size_ + 1 - mb_.offset.size());
      if (shuf_buf_ == 0) {
        Push(start_, len);
      } else {
        for (size_t i = start_; i < start_ + len; ++i) {
          mb_.Push(in_blk_[rdp_[i]]);
        }
      }
      start_ += len;
    }
    out_blk_ = mb_.GetBlock();
    return out_blk_.size > 0;
  }

  size_t BytesRead(void) const {
    return parser_ ? parser_->BytesRead() : buf_reader_->BytesRead();
  }

  const RowBlock<IndexType> &Value(void) const {
    return out_blk_;
  }

 private:
  void Push(size_t pos, size_t len) {
    if (!len) return;
    CHECK_LE(pos + len, in_blk_.size);
    RowBlock<IndexType> slice;
    slice.weight = NULL;
    slice.size = len;
    slice.offset  = in_blk_.offset + pos;
    slice.label   = in_blk_.label  + pos;
    slice.index   = in_blk_.index  + in_blk_.offset[pos];
    if (in_blk_.value) {
      slice.value = in_blk_.value  + in_blk_.offset[pos];
    } else {
      slice.value = NULL;
    }
    // LOG(INFO) << DebugStr(slice);
    mb_.Push(slice);
  }

  unsigned mb_size_, shuf_buf_;
  ParserImpl<IndexType> *parser_;

  size_t start_, end_;
  RowBlock<IndexType> in_blk_;
  RowBlockContainer<IndexType> mb_;
  RowBlock<IndexType> out_blk_;

  // random pertubation
  std::vector<unsigned> rdp_;
  MinibatchIter<IndexType>* buf_reader_;
};

}  // namespace data
}  // namespace dmlc

/**
 * @file   adfea_parser.h
 * @brief  parse adfea ctr data format
 */
#pragma once
#include <limits>
#include "data/row_block.h"
#include "data/parser.h"
#include "data/strtonum.h"
#include "base/crc32.h"
namespace dmlc {
namespace data {

/**
 * \brief criteo ctr dataset:
 * The columns are tab separeted with the following schema:
 *  <label> <integer feature 1> ... <integer feature 13>
 *  <categorical feature 1> ... <categorical feature 26>
 */
template <typename IndexType>
class AdfeaParser : public Parser<IndexType> {
 public:
  explicit AdfeaParser(InputSplit *source)
      : bytes_read_(0), source_(source) { }
  virtual ~AdfeaParser() {
    delete source_;
  }

  virtual void BeforeFirst(void) {
    source_->BeforeFirst();
  }
  virtual size_t BytesRead(void) const {
    return bytes_read_;
  }
  virtual bool ParseNext(std::vector<RowBlockContainer<IndexType> > *data) {
    InputSplit::Blob chunk;
    if (!source_->NextChunk(&chunk)) return false;

    CHECK(chunk.size != 0);
    bytes_read_ += chunk.size;
    data->resize(1);
    RowBlockContainer<IndexType>& blk = (*data)[0];
    blk.Clear();
    int i = 0;
    char *p = reinterpret_cast<char*>(chunk.dptr);
    char *end = p + chunk.size;
    while (isspace(*p) && p != end) ++p;
    while (p != end) {

      char *head = p;
      while (isdigit(*p) && p != end) ++p;
      CHECK_NE(head, p);

      if (*p == ':') {
        blk.index.push_back((IndexType)strtoull(head, NULL, 10));
        ++p;
        // skip the group id
        while (isdigit(*p) && p != end) ++p;
      } else {
        // skip the lineid and the first count
        if (i == 2) {
          i = 0;
          if (blk.label.size() != 0) {
            blk.offset.push_back(blk.index.size());
          }
          blk.label.push_back(*head == '1');
        } else {
          ++ i;
        }
      }

      while (isspace(*p) && p != end) ++p;
    }
    if (blk.label.size() != 0) {
      blk.offset.push_back(blk.index.size());
    }
    return true;
  }

 private:
  // number of bytes readed
  size_t bytes_read_;
  // source split that provides the data
  InputSplit *source_;
};

}  // namespace data
}  // namespace dmlc

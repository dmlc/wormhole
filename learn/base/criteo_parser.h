/**
 * @file   criteo_parser.h
 * @brief  parse criteo ctr data format
 */
#ifndef DMLC_DATA_CRITEO_PARSER_H_
#define DMLC_DATA_CRITEO_PARSER_H_
#include <limits>
#include "data/row_block.h"
#include "data/parser.h"
#include "data/strtonum.h"
#include "base/crc32.h"
#include "proto/data_format.pb.h"
#include "dmlc/recordio.h"
namespace dmlc {
namespace data {

/**
 * \brief criteo ctr dataset:
 * The columns are tab separeted with the following schema:
 *  <label> <integer feature 1> ... <integer feature 13>
 *  <categorical feature 1> ... <categorical feature 26>
 */
template <typename IndexType>
class CriteoParser : public ParserImpl<IndexType> {
 public:
  explicit CriteoParser(InputSplit *source)
      : bytes_read_(0), source_(source) { }
  virtual ~CriteoParser() {
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
    char *p = reinterpret_cast<char*>(chunk.dptr);
    char *end = p + chunk.size;
    data->resize(1);
    RowBlockContainer<IndexType>& blk = (*data)[0];
    blk.Clear();
    IndexType kmax = std::numeric_limits<IndexType>::max();
    IndexType itv = kmax / 13 + 1;
    char *pp = p;
    while (p != end) {
      while (*p == '\r' || *p == '\n') ++p;
      if (p == end) break;

      // parse label
      pp = Find(p, end, '\t');
      CHECK_NE(pp, p) << "cannot parse criteo test data";
      blk.label.push_back(atof(p));
      p = pp + 1;

      // parse inter feature
      for (IndexType i = 0,  os = 0; i < 13; ++i, os += itv) {
        pp = Find(p, end, '\t');
        CHECK_NOTNULL(pp);
        if (pp > p) {
          blk.index.push_back(atol(p) + os);
        }
        p = pp + 1;
      }

      // parse categorty feature
      for (int i = 0; i < 26; ++i) {
        if (isspace(*p)) {
          ++ p; continue;
        }
        pp = p + 8; CHECK(isspace(*pp)) << *pp;
        size_t len = pp - p;
        if (len) blk.index.push_back(CRC32HW(p, len));
        p = pp + 1;
      }
      blk.offset.push_back(blk.index.size());
    }
    return true;
  }

 private:

  // implement strchr
  inline char* Find(char* p, char* end, int c) {
    while (p != end && *p != c) ++p;
    return p;
  }

  // number of bytes readed
  size_t bytes_read_;
  // source split that provides the data
  InputSplit *source_;
};

template <typename IndexType>
class CriteoRecParser : public ParserImpl<IndexType> {
 public:
  CriteoRecParser(InputSplit *source)
      : bytes_read_(0), source_(source) { }
  virtual ~CriteoRecParser() {
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

    IndexType itv = std::numeric_limits<IndexType>::max() / 13 + 1;
    std::string str;
    linear::Criteo pb;
    InputSplit::Blob rec;
    RecordIOChunkReader reader(chunk);
    while (reader.NextRecord(&rec)) {
      CHECK(pb.ParseFromArray(rec.dptr, rec.size));

      // label
      blk.label.push_back(pb.label());

      // parse categorty feature
      uint32_t miss_int = pb.miss_int();
      int k = 0;
      for (IndexType i = 0,  os = 0; i < 13; ++i, os += itv) {
        if (miss_int & (1<<i)) continue;
        CHECK_LT(k, pb.dint_size());
        blk.index.push_back(pb.dint(k) + os);
        ++ k;
      }
      CHECK_EQ(k, pb.dint_size());

      uint32_t miss_cat = pb.miss_cat();
      k = 0;
      for (int i = 0; i < 26; ++i) {
        if (miss_cat & (1<<i)) continue;
        CHECK_LT(k, pb.dcat_size());
        blk.index.push_back(pb.dcat(k));
        ++ k;
      }
      CHECK_EQ(k, pb.dcat_size());

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


#endif /* DMLC_DATA_CRITEO_PARSER_H_ */

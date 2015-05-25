/**
 * @file   criteo_rec_parser.h
 * @brief  parse criteo ctr data format
 */
#pragma once
#include <limits>
#include "data/row_block.h"
#include "data/parser.h"
#include "data/strtonum.h"
#include "proto/data_format.pb.h"
#include "dmlc/recordio.h"
namespace dmlc {
namespace data {

template <typename IndexType>
class CriteoRecParser : public Parser<IndexType> {
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

/**
 * @file   adfea_rec_parser.h
 * @brief  parse adfea ctr data format
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
class AdfeaRecParser : public Parser<IndexType> {
 public:
  AdfeaRecParser(InputSplit *source)
      : bytes_read_(0), source_(source) { }
  virtual ~AdfeaRecParser() {
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

    std::string str;
    linear::Adfea pb;
    InputSplit::Blob rec;
    RecordIOChunkReader reader(chunk);
    while (reader.NextRecord(&rec)) {
      CHECK(pb.ParseFromArray(rec.dptr, rec.size));
      blk.label.push_back(pb.label());
      for (int i = 0; i < pb.feaid_size(); ++i) {
        blk.index.push_back(pb.feaid(i));
      }
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

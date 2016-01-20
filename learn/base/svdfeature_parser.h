/* 
 * File:   svdfeature_parser.h
 * Author: hexi
 *
 * Created on 2015年12月28日, 下午3:24
 */

#ifndef SVDFEATURE_PARSER_H
#define	SVDFEATURE_PARSER_H
#include <iostream>
#include <limits>
#include <city.h>
#include "data/row_block.h"
#include "data/parser.h"
#include "data/strtonum.h"
#include "dmlc/recordio.h"
namespace dmlc {
namespace data {

/**
 * \brief criteo ctr dataset:
 * The data use the following schema:
 *  label:weight |g g1 g2 ... |u u1 u2 ... |i i1 i2 ...
 */
template <typename IndexType>
class SvdFeaParser : public ParserImpl<IndexType> {
 public:
  explicit SvdFeaParser(InputSplit *source)
      : bytes_read_(0), source_(source) {
  }
  virtual ~SvdFeaParser() {
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
    
    char* lbegin = p;
    char* lend = lbegin;
    char* head = NULL;
    real_t label;
    real_t weight;
    IndexType idx;
    int shift_bits = (sizeof(IndexType) == 8 ? 62: 30);
    IndexType kt = 2; //default=2 0:user 1:item 2:global 
    while (lbegin != end) {
        // get line end
        lend = lbegin + 1;
        while (lend != end && *lend != '\n' && *lend != '\r') ++lend;
        if(lend == end) {
            break;
        }
        p = lbegin;
        while (p != lend && isspace(*p)) ++p;
        if(p == lend) {
            lbegin = lend;
            continue;
        }
        
        head = p;
        label = static_cast<float>(strtod(head, &p));
        weight = 1.0f;
        if(p != lend && *p == ':'){
            head = ++p;
            weight = static_cast<float>(strtod(head, &p));
        }
        blk.label.push_back(label);
        blk.weight.push_back(weight);
        
        while(p != lend) {
            while (p != lend && isspace(*p)) ++p;
            if (p == lend) break;
            kt = 2;
            if(*p == '|') {
                if (++p == lend) break;
                CHECK((*p=='g'||*p=='u'||*p=='i')) << "invalid data format";
                kt = (*p == 'g' ? 2 : (*p == 'u' ? 0 : 1));
                ++p;
            }
            
            while(p != lend) {
                while (p != lend && isspace(*p)) ++p;
                if (p == lend || *p == '|') break;
                head = p;
                while ( p != lend && isdigit(*p)) ++p;
                if (*p == ':') {
                    idx = static_cast<IndexType>(atol(head));
                    idx = idx & ((static_cast<IndexType>(1) << shift_bits) - 1);
                    idx |= (kt << shift_bits);
                    blk.index.push_back(idx);
                    blk.value.push_back(atof(p + 1));
                } else {
                    idx = static_cast<IndexType>(atol(head));
                    idx = idx & ((static_cast<IndexType>(1) << shift_bits) - 1);
                    idx |= (kt << shift_bits);
                    blk.index.push_back(idx);
                    blk.value.push_back(1.0);
                }
                while (p != lend && !isspace(*p) ) ++p;
            }
        }
        blk.offset.push_back(blk.index.size());
        lbegin = lend;
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


#endif	/* SVDFEATURE_PARSER_H */


/**
 * @file   minibatch_iter.h
 * @brief  A minibatch iterator
 */
#pragma once
#include <dmlc/logging.h>
#include <dmlc/io.h>
namespace dmlc {
namespace data {

// TODO
template<typename IndexType>
class MinibatchIter {
 public:
  MinibatchIter(const char* uri, unsigned part_idex, usigned num_parts,
                const char* type, unsighed minibatch_size)
      : mb_(minibatch_size) {

    io::URISpec spec(uri_, part_index, num_parts);
    // create parser
    if (!strcmp(type, "libsvm")) {
      parser = new LibSVMParser(InputSplit::Create(spec.uri.c_str(),
                                                   part_index, num_parts,
                                                   "text"), 16);
    } else {
      LOG(FATAL) << "unknown datatype " << type;
    }
	parser = new ThreadedParser(parser);
  }
  virtual ~MinibatchIter() {
    delete parser_;
  }

  void BeforeFirst(void) {

  }
  bool Next(void) {
    return true;
  }

  const RowBlock<IndexType> &Value(void) const {
    return block_;
  }

 private:
  unsighed mb_;
  Parser *parser_;

  RowBlock<size_t> block_;
};
}  // namespace data
}  // namespace dmlc

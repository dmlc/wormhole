/**
 * @file   convert.cc
 * @brief  Convert data from one format to another format
 */

#include <city.h>
#include "gflags/gflags.h"
#include "dmlc/recordio.h"
#include "dmlc/data.h"
#include "dmlc/logging.h"
#include "data/libsvm_parser.h"
#include "base/adfea_parser.h"
#include "base/criteo_parser.h"
#include "base/compressed_row_block.h"

DEFINE_string(data_in, "stdin", "input filename name or stdin");
DEFINE_string(data_out, "stdout", "output filename name or stdout");
DEFINE_string(format_in, "libsvm", "input data format");
DEFINE_string(format_out, "crb", "output data format");
DEFINE_int32(part_size, -1, "split the output into multiple parts, \
with each part <= part_size MB");

int main(int argc, char *argv[]) {
  using namespace dmlc;
  using namespace dmlc::data;
  InitLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  // input
  // using IndexType = uint32_t;
  using IndexType = uint64_t;
  InputSplit* in = CHECK_NOTNULL(
      InputSplit::Create(FLAGS_data_in.c_str(), 0, 1, "text"));
  in->HintChunkSize(1<<22);  // 4MB chuck
  ParserImpl<IndexType> * parser = NULL;
  auto type = FLAGS_format_in;
  if (type == "libsvm") {
    parser = new LibSVMParser<IndexType>(in, 1);
  } else if (type == "criteo") {
    parser = new CriteoParser<IndexType>(in, true);
  } else if (type == "criteo_test") {
    parser = new CriteoParser<IndexType>(in, false);
  } else if (type == "adfea") {
    parser = new AdfeaParser<IndexType>(in);
  } else {
    LOG(FATAL) << "unknown format " << type;
  }
  parser = new ThreadedParser<IndexType>(parser);

  // output
  size_t part_size = (size_t)FLAGS_part_size;
  size_t nwrite = (size_t)-1;
  int ipart = 0;
  type = FLAGS_format_out;
  Stream *out = NULL;
  RecordIOWriter* crb_writer = NULL;
  ostream* libsvm_writer = NULL;

  char outfile[1000];

  // convert
  parser->BeforeFirst();

  std::string str;
  CompressedRowBlock cblk;
  while (parser->Next()) {
    if (nwrite * 1000000 >= part_size) {
      if (part_size == (size_t)-1) {
        snprintf(outfile, 1000, "%s", FLAGS_data_out.c_str());
      } else {
        snprintf(outfile, 1000, "%s-part_%02d", FLAGS_data_out.c_str(), ipart);
        ipart ++;
      }
      delete libsvm_writer;
      delete crb_writer;
      delete out;
      out = CHECK_NOTNULL(Stream::Create(outfile, "wb"));
      nwrite = 0;

      if (type == "libsvm") {
        libsvm_writer = new ostream(out);
      } else if (type == "crb") {
        crb_writer = new RecordIOWriter(out);
      } else {
        LOG(FATAL) << "unknow output format: " << type;
      }
    }

    if (type == "libsvm") {
      auto blk = parser->Value();
      size_t last = libsvm_writer->bytes_written();
      for (size_t i = 0; i < blk.size; ++i) {
        *libsvm_writer << blk.label[i] << " ";
        for (size_t j = blk.index[i]; j < blk.index[i+1]; ++j) {
          *libsvm_writer << blk.index[j];
          if (blk.value) *libsvm_writer << ":" << blk.value[j];
        }
        *libsvm_writer << "\n";
      }
      nwrite += libsvm_writer->bytes_written() - last;
    } else if (type == "crb") {
      cblk.Compress(parser->Value(), &str);
      crb_writer->WriteRecord(str);
      nwrite += str.size();
    }
  }

  delete in;
  delete out;
  delete libsvm_writer;
  delete crb_writer;

  return 0;
}

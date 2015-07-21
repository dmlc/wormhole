/**
 * @file   text2rec.cc
 * @brief  Convert from text to recrodio for different data format
 */

#include <city.h>
#include "dmlc/recordio.h"
#include "dmlc/data.h"
#include "dmlc/logging.h"
#include "data/libsvm_parser.h"
#include "base/adfea_parser.h"
#include "base/criteo_parser.h"
#include "base/compressed_row_block.h"

int main(int argc, char *argv[]) {
  using namespace dmlc;
  using namespace dmlc::data;
  InitLogging(argv[0]);
  if (argc < 4) {
    printf("Usage: input output format [part_size] \n");
    printf(" - input: a input file name or stdin\n");
    printf(" - output: a output file name or stdout\n");
    printf(" - format: libsvm, criteo, adfea, ... \n");
    printf(" - part_size: split the output into multiple parts, \
with each part <= part_size MB \n");
    return 0;
  }

  // input
  // using IndexType = uint32_t;
  using IndexType = uint64_t;
  InputSplit* in = CHECK_NOTNULL(InputSplit::Create(argv[1], 0, 1, "text"));
  in->HintChunkSize(1<<22);  // 4MB chuck
  ParserImpl<IndexType> * parser = NULL;
  const char* type = argv[3];
  if (!strcmp(type, "libsvm")) {
    parser = new LibSVMParser<IndexType>(in, 1);
  } else if (!strcmp(type, "criteo")) {
    parser = new CriteoParser<IndexType>(in, true);
  } else if (!strcmp(type, "criteo_test")) {
    parser = new CriteoParser<IndexType>(in, false);
  } else if (!strcmp(type, "adfea")) {
    parser = new AdfeaParser<IndexType>(in);
  } else {
    LOG(FATAL) << "unknown format " << type;
  }
  parser = new ThreadedParser<IndexType>(parser);

  // output


  std::string outbase = argv[2];

  size_t part_size = (size_t)-1;
  size_t nwrite = (size_t)-1;
  int ipart = 0;
  if (argc > 4) part_size = atoi(argv[4]) * 1000000;


  Stream *out = NULL;
  RecordIOWriter* writer = NULL;
  char outfile[1000];

  // convert
  parser->BeforeFirst();

  std::string str;
  CompressedRowBlock cblk;
  while (parser->Next()) {

    if (nwrite >= part_size) {
      if (part_size == (size_t)-1) {
        snprintf(outfile, 1000, "%s", argv[2]);
      } else {
        snprintf(outfile, 1000, "%s-part_%02d", argv[2], ipart);
        ipart ++;
      }
      delete out;
      delete writer;
      out = CHECK_NOTNULL(Stream::Create(outfile, "wb"));
      nwrite = 0;
      writer = new RecordIOWriter(out);
    }

    cblk.Compress(parser->Value(), &str);
    writer->WriteRecord(str);
    nwrite += str.size();
  }

  delete in;
  delete out;
  delete writer;

  return 0;
}

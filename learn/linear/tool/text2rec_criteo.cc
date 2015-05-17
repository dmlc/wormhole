/**
 * @file   text2rec_criteo.cc
 * @brief  Convert from text to recrodio for criteo dataset
 */
#include <city.h>
#include "dmlc/recordio.h"
#include "dmlc/data.h"
#include "dmlc/logging.h"
#include "proto/criteo.pb.h"
#include "data/strtonum.h"
#include "base/crc32.h"
// implement strchr
inline char* Find(char* p, char* end, int c) {
  while (p != end && *p != c) ++p;
  return p;
}

int main(int argc, char *argv[]) {
  using namespace dmlc;
  InitLogging(argv[0]);
  if (argc < 3) {
    printf("Usage: input output\n");
    return 0;
  }

  Stream *out = CHECK_NOTNULL(Stream::Create(argv[2], "wb"));
  RecordIOWriter writer(out);

  InputSplit* in = CHECK_NOTNULL(InputSplit::Create(argv[1], 0, 1, "text"));
  in->BeforeFirst();
  InputSplit::Blob chunk;
  linear::Criteo pb;
  std::string pb_str;
  size_t num = 0;
  while (in->NextChunk(&chunk)) {
    char *p = reinterpret_cast<char*>(chunk.dptr);
    char *end = p + chunk.size;

    char *pp = p;
    while (p != end) {
      while (*p == '\r' || *p == '\n') ++p;
      if (p == end) break;
      pb.Clear();

      // parse label
      pp = Find(p, end, '\t');
      CHECK_NE(pp, p) << "cannot parse criteo test data";
      pb.set_label(atol(p));
      p = pp + 1;

      // parse inter feature
      uint32_t miss_int = 0;
      for (int i = 0; i < 13; ++i) {
        pp = CHECK_NOTNULL(Find(p, end, '\t'));
        if (pp > p) {
          pb.add_dint(atol(p));
        } else {
          miss_int |= 1 << i;
        }
        p = pp + 1;
      }
      pb.set_miss_int(miss_int);

      // parse categorty feature
      uint32_t miss_cat = 0;
      for (int i = 0; i < 26; ++i) {
        if (isspace(*p)) {
          ++ p;
          miss_cat |= 1 << i;
          continue;
        }
        pp = p + 8; CHECK(isspace(*pp)) << *pp;
        size_t len = pp - p;
        if (len) {
          pb.add_dcat(CityHash32(p, len));
        } else {
          miss_cat |= 1 << i;
        }
        p = pp + 1;
      }
      pb.set_miss_cat(miss_cat);

      CHECK(pb.SerializeToString(&pb_str));
      writer.WriteRecord(pb_str);

      num ++;
    }
    // LOG(ERROR) << num;
  }

  delete out;
  delete in;

  return 0;
}

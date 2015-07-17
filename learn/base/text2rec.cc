/**
 * @file   text2rec.cc
 * @brief  Convert from text to recrodio for different data format
 */

#include <city.h>
#include "dmlc/recordio.h"
#include "dmlc/data.h"
#include "dmlc/logging.h"
#include "proto/data_format.pb.h"
#include "data/strtonum.h"

// implement strchr
inline char* Find(char* p, char* end, int c) {
  while (p != end && *p != c) ++p;
  return p;
}

void ParseCriteo(char* p, char* end, dmlc::RecordIOWriter* writer) {
  dmlc::data::Criteo pb;
  std::string pb_str;
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
    for (int i = 0; i < 13; ++i) {
      pp = CHECK_NOTNULL(Find(p, end, '\t'));
      if (pp > p) {
        pb.add_feaid((CityHash32(p, pp-p)<<6)+i);
      }
      p = pp + 1;
    }

    // parse categorty feature
    for (int i = 0; i < 26; ++i) {
      if (isspace(*p)) {
        ++ p;
        continue;
      }
      pp = p + 8; CHECK(isspace(*pp)) << *pp;
      size_t len = pp - p;
      if (len) {
        pb.add_feaid((CityHash32(p, len)<<6)+13+i);
      }
      p = pp + 1;
    }
    CHECK(pb.SerializeToString(&pb_str));
    writer->WriteRecord(pb_str);
  }
}

void ParseAdfea(char* p, char* end, dmlc::RecordIOWriter* writer) {
  dmlc::data::Adfea pb;
  std::string pb_str;
  int i = 0;
  while (isspace(*p) && p != end) ++p;
  while (p != end) {
    char *head = p;
    while (isdigit(*p) && p != end) ++p;
    CHECK_NE(head, p);

    if (*p == ':') {
      pb.add_feaid(strtoull(head, NULL, 10));
      ++p;
      // skip the group id
      while (isdigit(*p) && p != end) ++p;
    } else {
      // skip the lineid and the first count
      if (i == 2) {
        if (pb.has_label()) {
          CHECK(pb.SerializeToString(&pb_str));
          writer->WriteRecord(pb_str);
          pb.Clear();
        }
        pb.set_label(*head == '1');
        i = 0;
      } else {
        ++ i;
      }
    }
    while (isspace(*p) && p != end) ++p;
  }
  if (pb.has_label()) {
    CHECK(pb.SerializeToString(&pb_str));
    writer->WriteRecord(pb_str);
  }
}

int main(int argc, char *argv[]) {
  using namespace dmlc;
  InitLogging(argv[0]);
  if (argc < 4) {
    printf("Usage: input output format\n");
    printf("supported format: criteo, adfea\n");
    return 0;
  }

  Stream *out = CHECK_NOTNULL(Stream::Create(argv[2], "wb"));
  RecordIOWriter writer(out);

  InputSplit* in = CHECK_NOTNULL(InputSplit::Create(argv[1], 0, 1, "text"));
  in->BeforeFirst();
  InputSplit::Blob chunk;

  while (in->NextChunk(&chunk)) {
    char *start = reinterpret_cast<char*>(chunk.dptr);
    char *end = start + chunk.size;

    if (!strcmp(argv[3], "criteo")) {
      ParseCriteo(start, end, &writer);
    } else if (!strcmp(argv[3], "adfea")) {
      ParseAdfea(start, end, &writer);
    } else {
      LOG(FATAL) << "unknow format " << argv[3];
    }
  }
  delete out;
  delete in;

  return 0;
}

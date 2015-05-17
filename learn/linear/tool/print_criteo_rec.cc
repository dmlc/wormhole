#include <stdio.h>
#include <string>
#include "dmlc/logging.h"
#include "dmlc/recordio.h"
#include "proto/criteo.pb.h"
int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: input num_record\n");
    return 0;
  }

  using namespace dmlc;
  InitLogging(argv[0]);

  Stream *in = CHECK_NOTNULL(Stream::Create(argv[1], "rb"));
  RecordIOReader reader(in);

  std::string str;
  linear::Criteo pb;
  for (int i = 0; i < atoi(argv[2]); ++i) {
    CHECK(reader.NextRecord(&str));
    CHECK(pb.ParseFromString(str));
    printf("%s\n", pb.ShortDebugString().c_str());
  }

  return 0;
}

#include <stdio.h>
#include <string>
#include "dmlc/logging.h"
#include "dmlc/recordio.h"
#include "proto/data_format.pb.h"
int main(int argc, char *argv[]) {
  if (argc < 4) {
    printf("Usage: input num_record format\n");
    return 0;
  }

  using namespace dmlc;
  InitLogging(argv[0]);

  Stream *in = CHECK_NOTNULL(Stream::Create(argv[1], "rb"));
  RecordIOReader reader(in);

  std::string str;
  for (int i = 0; i < atoi(argv[2]); ++i) {
    CHECK(reader.NextRecord(&str));
    if (!strcmp(argv[3], "criteo")) {
      linear::Criteo pb;
      CHECK(pb.ParseFromString(str));
      printf("%s\n", pb.ShortDebugString().c_str());
    } else if (!strcmp(argv[3], "adfea")) {
      linear::Adfea pb;
      CHECK(pb.ParseFromString(str));
      printf("%s\n", pb.ShortDebugString().c_str());
    } else {
      LOG(FATAL) << "unknow format " << argv[3];
    }
  }

  return 0;
}

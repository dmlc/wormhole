/**
 * @file   arg2proto.h
 * @brief  Parse the arg to a proto
 */
#pragma once

#include "dmlc/logging.h"
#include "dmlc/config.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
namespace dmlc {

inline void Arg2Proto(int argc, char *argv[], google::protobuf::Message* proto) {
  CHECK_GE(argc, 2);
  std::ifstream in(argv[1]);
  CHECK(in.good()) << "failed to open configure file: " << argv[1];
  Config parser(in, true);
  CHECK(google::protobuf::TextFormat::ParseFromString(
      parser.ToProtoString(), proto));
}

}  // namespace dmlc

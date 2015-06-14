/**
 * @file   arg_parser.h
 * @brief  A simple arg parser
 */
#pragma once
#include <string>
#include "dmlc/io.h"
#include "dmlc/logging.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
namespace dmlc {

class ArgParser {
 public:
  ArgParser() { }
  ~ArgParser() { }

  void ReadFile(const char* const filename) {
    Stream *fs = Stream::Create(filename, "r");
    char buf[1000];
    while (true) {
      size_t r = fs->Read(buf, 1000);
      file_.append(buf, r);
      if (!r) break;
    }
    CHECK(!file_.empty()) << "nothing read from " << filename;
  }

  void ReadArgs(int argc, char* argv[]) {
    for (int i = 0; i < argc; ++i) {
      args_.append(argv[i]);
      args_.append(" ");
    }
  }

  void ParseToProto(google::protobuf::Message* proto) {
    if (!file_.empty()) {
      CHECK(google::protobuf::TextFormat::ParseFromString(
          ReplaceStr(file_), proto));
    }
    if (!args_.empty()) {
      CHECK(google::protobuf::TextFormat::MergeFromString(
          ReplaceStr(args_), proto)) << ReplaceStr(args_);
    }
  }
 private:
  /*! replace = to : */
  std::string ReplaceStr(const std::string& str) {
    std::string out = str;
    int in_quote = 0;
    for (size_t i = 0; i < out.size(); ++i) {
      char c = out[i];
      if (c == '=' && in_quote == 0) out[i] = ':';
      if (c == '\'' || c == '"') {
        // TODO does not support recurcise quote or \"
        in_quote = 1 - in_quote;
      }
    }
    return out;
  }
  std::string file_;
  std::string args_;

};

}  // namespace dmlc

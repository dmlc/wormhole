#pragma once
#include "io/filesys.h"
#include "dmlc/logging.h"
#include <vector>
#include <string>
#include <regex>
namespace dmlc {

// match files by regex pattern
// such as s3://my_path/part-.*
inline void MatchFile(const std::string& pattern,
                      std::vector<std::string>* matched) {
  // get the path
  size_t pos = pattern.find_last_of("/\\");
  std::string path = "./";
  if (pos != std::string::npos) path = pattern.substr(0, pos);

  // find all files
  dmlc::io::URI path_uri(path.c_str());
  dmlc::io::FileSystem *fs =
      dmlc::io::FileSystem::GetInstance(path_uri.protocol);
  std::vector<io::FileInfo> info;
  fs->ListDirectory(path_uri, &info);

  // store all matached files
  std::regex pat;
  try {
    std::string file =
        pos == std::string::npos ? pattern : pattern.substr(pos+1);
    file = ".*" + file;
    pat = std::regex(".*"+file);
  } catch (const std::regex_error& e) {
    LOG(FATAL) << pattern << " is not valid regex, or unsupported regex"
               << ". you may try gcc>=4.9 or llvm>=3.4";
  }

  CHECK_NOTNULL(matched);
  for (size_t i = 0; i < info.size(); ++i) {
    std::string file = info[i].path.str();
    if (!std::regex_match(file, pat)) {
      continue;
    }
    matched->push_back(file);
  }
}
} // namespace dmlc

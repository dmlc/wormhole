/**
 * @file   workload.h
 * @brief  Workload
 */
#pragma once
#include <string>
#include "dmlc/io.h"

namespace dmlc {

struct Workload : public Serializable {
  enum Type { TRAIN, VAL } type;
  int data_pass;
  struct File {
    std::string filename;
    std::string format;  // data format
    int n = 1;  // num of part
    int k = 0;  // the k-th parf

    std::string ShortDebugString() const {
      std::string ret
          = filename + " " + std::to_string(k) + " / " + std::to_string(n);
      return ret;
    }
  };
  std::vector<File> file;

  virtual void Load(Stream* fi) {
    fi->Read(&type, sizeof(type));
    fi->Read(&data_pass, sizeof(data_pass));
    size_t num;
    fi->Read(&num, sizeof(num));
    for (size_t i = 0; i < num; ++i) {
      File f;
      fi->Read(&f.filename);
      fi->Read(&f.format);
      fi->Read(&f.n, sizeof(f.n));
      fi->Read(&f.k, sizeof(f.k));
      file.push_back(f);
    }
  }

  virtual void Save(Stream *fo) const {
    fo->Write(&type, sizeof(type));
    fo->Write(&data_pass, sizeof(data_pass));
    size_t num = file.size();
    fo->Write(&num, sizeof(num));
    for (const auto& f : file) {
      fo->Write(f.filename);
      fo->Write(f.format);
      fo->Write(&f.n, sizeof(f.n));
      fo->Write(&f.k, sizeof(f.k));
    }
  }
};


}  // namespace dmlc

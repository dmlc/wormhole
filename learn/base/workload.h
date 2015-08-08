/**
 * @file   workload.h
 * @brief  Workload
 */
#pragma once
#include <string>
#include <sstream>
#include "dmlc/io.h"

namespace dmlc {

/**
 * \brief a workload the scheduler assigns to workers
 */
struct Workload : public Serializable {
  enum Type { TRAIN, VAL } type;
  /// \brief current pass of data, start from 1
  int data_pass;

  struct File {
    /// \brief filename
    std::string filename;
    /// \brief data format: libsvm, crb, ...
    std::string format;
    /// \brief number of virtual parts of this file
    int n = 1;
    /// \brief the workload only needs to process the k-th part
    int k = 0;

    std::string ShortDebugString() const {
      std::string ret
          = filename + " " + std::to_string(k) + " / " + std::to_string(n);
      return ret;
    }
  };

  /// \brief files needed to be processed
  std::vector<File> file;

  std::string ShortDebugString() const {
    std::stringstream ss;
    ss << "iter = " << data_pass << ", "
       << (type == TRAIN ? "training," : "validation,");
    for (const auto& f : file) {
      ss << " " << f.ShortDebugString();
    }
    return ss.str();
  }

  /// \brief empty workload
  bool Empty() { return file.size() == 0; }

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

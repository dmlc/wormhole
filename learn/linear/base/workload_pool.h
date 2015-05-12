#pragma once
#include "proto/sys.pb.h"
#include "io/filesys.h"
#include "dmlc/logging.h"
#include <vector>
#include <list>
#include <regex>
namespace dmlc {
namespace linear {

/**
 * @brief A thread-safe workload pool
 */
class WorkloadPool {
 public:
  WorkloadPool() { }
  ~WorkloadPool() { }

  /**
   * @brief init the workload
   *
   * @param files s3://my_path/part-.*
   * @param npart divide one file into npart
   * @param nconsumer
   */
  void Add(const std::string& files, int npart,
           int nconsumer = 0, Workload::Type type = Workload::TRAIN) {
    std::lock_guard<std::mutex> lk(mu_);
    // get the path
    size_t pos = files.find_last_of("/\\");
    std::string path = "./";
    if (pos != std::string::npos)
      path = files.substr(0, pos);

    // find all files
    dmlc::io::URI path_uri(path.c_str());
    dmlc::io::FileSystem *fs = dmlc::io::FileSystem::GetInstance(path_uri.protocol);
    std::vector<io::FileInfo> info;
    fs->ListDirectory(path_uri, &info);

    // store all matached files
    std::regex pattern;
    try {
      pattern = std::regex(files);
    } catch (const std::regex_error& e) {
      LOG(FATAL) << files << " is not valid regex, or unsupported regex"
                 << ". you may try gcc>=4.9 or llvm>=3.4";
    }
    for (size_t i = 0; i < info.size(); ++i) {
      std::string file = info[i].path.str();
      if (!std::regex_match(file, pattern)) continue;

      LOG(INFO) << "found file: " << file;
      for (int j = 0; j < npart; ++j) {
        File f;
        f.set_file(file);
        f.set_n(npart);
        f.set_k(j);
        remain_.push_back(f);
      }
    }

    type_ = type;
    num_ = nconsumer == 0 ? 0 : remain_.size() / nconsumer;
  }

  void Clear() {
    std::lock_guard<std::mutex> lk(mu_);
    remain_.clear();
    assigned_.clear();
    num_ = 0;
  }

  // get one to id when nconsumer == 0
  // divide the workload into nconsumer part, give one to id
  void Get(const std::string& id, Workload* wl) {
    std::lock_guard<std::mutex> lk(mu_);
    wl->Clear();
    if (num_ == 0) {
      GetOne(id, wl);
    } else {
      for (int i = 0; i < num_; ++i) {
        GetOne(id, wl);
      }
    }
  }

  // id dies
  void Reset(const std::string& id) { Set(id, false); }

  // id finished the workload it got before
  void Finish(const std::string& id) { Set(id, true); }

  bool IsFinished() {
    std::lock_guard<std::mutex> lk(mu_);
    return remain_.empty() && assigned_.empty();
  }

 private:
  void Set(const std::string& id, bool del) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = assigned_.begin();
    while (it != assigned_.end()) {
      if (it->first == id) {
        if (!del) {
          LOG(INFO) << id << " failed to finish workload " << it->second.ShortDebugString();
          remain_.push_front(it->second);
        } else {
          LOG(INFO) << id << " finished workload " << it->second.ShortDebugString();
        }
        it = assigned_.erase(it);
      } else {
        ++ it;
      }
    }

  }

  void GetOne(const std::string& id, Workload* wl) {
    if (remain_.empty()) return;
    wl->add_file()->CopyFrom(remain_.front());
    wl->set_type(type_);
    assigned_.push_back(std::make_pair(id, remain_.front()));
    LOG(INFO) << "assign " << id << " workload "
              << remain_.front().ShortDebugString();
    remain_.pop_front();
  }

  Workload::Type type_;
  int num_;
  // std::vector<File> files_;
  std::list<File> remain_;
  std::list<std::pair<std::string, File>> assigned_;
  std::mutex mu_;
};

}  // namespace linear
}  // namespace dmlc

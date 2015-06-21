#pragma once
#include "proto/workload.pb.h"
#include "io/filesys.h"
#include "dmlc/logging.h"
#include <vector>
#include <list>
#include <regex>
namespace dmlc {

/**
 * @brief A thread-safe workload pool
 */
class WorkloadPool {
 public:
  WorkloadPool() {
    straggler_killer_ = new std::thread([this]() {
        while (!IsFinished()) {
          RemoveStraggler();
          sleep(1);
        }
      });
  }
  ~WorkloadPool() {
    straggler_killer_->join();
    delete straggler_killer_;
  }

  /**
   * @brief add the workload
   *
   * @param files s3://my_path/part-.*
   * @param format libsvm, criteo, ...
   * @param npart divide one file into npart
   * @param nconsumer
   */
  void Add(const std::string& files, const std::string& format, int npart,
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
      std::string file = pos == std::string::npos ? files : files.substr(pos+1);
      file = ".*" + file;
      LOG(INFO) << "match files by: " << file;
      pattern = std::regex(".*"+file);
    } catch (const std::regex_error& e) {
      LOG(FATAL) << files << " is not valid regex, or unsupported regex"
                 << ". you may try gcc>=4.9 or llvm>=3.4";
    }

    LOG(INFO) << "found " << info.size() << " files";
    for (size_t i = 0; i < info.size(); ++i) {
      std::string file = info[i].path.str();
      if (!std::regex_match(file, pattern)) {
        continue;
      }
      LOG(INFO) << "matched file: " << file;
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
    format_ = format;
  }

  void Clear() {
    std::lock_guard<std::mutex> lk(mu_);
    remain_.clear();
    assigned_.clear();
    num_ = 0;
    num_finished_ = 0;
  }

  void ClearRemain() {
    std::lock_guard<std::mutex> lk(mu_);
    remain_.clear();
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

  int num_finished() { return num_finished_; }
  int num_assigned() { return assigned_.size(); }

 private:
  void Set(const std::string& id, bool del) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = assigned_.begin();
    while (it != assigned_.end()) {
      if (it->node == id) {
        if (!del) {
          LOG(INFO) << id << " failed to finish workload " << it->file.ShortDebugString();
          remain_.push_front(it->file);
        } else {
          double time = GetTime() - it->start_time;
          LOG(INFO) << id << " finished " << it->file.ShortDebugString()
                    << " in " << time << " sec.";
          time_.push_back(time);
          ++ num_finished_;
        }
        it = assigned_.erase(it);

        std::string nd;
        int k = 0;
        for (const auto& it2 : assigned_) {
          if (++k > 5) break;
          nd += ", " + it2.node;
        }
        LOG(INFO) << assigned_.size() << " files are on processing by " << nd << "...";
      } else {
        ++ it;
      }
    }
  }

  void GetOne(const std::string& id, Workload* wl) {
    if (remain_.empty()) return;
    wl->add_file()->CopyFrom(remain_.front());
    wl->set_type(type_);
    wl->set_format(format_);
    ActiveTask task;
    task.node = id;
    task.file = remain_.front();
    task.start_time = GetTime();
    assigned_.push_back(task);
    LOG(INFO) << "assign " << id << " workload "
              << remain_.front().ShortDebugString();
    remain_.pop_front();
  }

  void RemoveStraggler() {
    std::lock_guard<std::mutex> lk(mu_);
    if (time_.size() < 10); return;
    double mean = 0;
    for (double t : time_) mean += t;
    mean /= time_.size();
    double cur_t = GetTime();
    auto it = assigned_.begin();
    while (it != assigned_.end()) {
      double t = cur_t - it->start_time;
      if (t > mean * 3) {
        LOG(INFO) << it->node << " is processing "
                  << it->file.ShortDebugString() << " for " << t
                  << " sec, which is much longer than the average time "
                  << mean << " sec. reassign this workload to other nodes";
        remain_.push_front(it->file);
        it = assigned_.erase(it);
      } else {
        ++ it;
      }
    }
  }

  std::string format_;
  Workload::Type type_;
  int num_;

  int num_finished_ = 0;
  std::list<File> remain_;
  struct ActiveTask {
    std::string node;
    File file;
    double start_time;
  };
  std::list<ActiveTask> assigned_;

  // process time
  std::vector<double> time_;
  std::mutex mu_;
  std::thread* straggler_killer_;
};

}  // namespace dmlc

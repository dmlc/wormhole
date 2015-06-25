#pragma once
#include "dmlc/logging.h"
#include "dmlc/timer.h"
#include "base/workload.h"
#include "base/match_file.h"
#include <sstream>
#include <vector>
#include <list>
namespace dmlc {

/**
 * @brief A thread-safe workload pool
 */
class WorkloadPool {
 public:
  WorkloadPool() {
    straggler_killer_ = new std::thread([this]() {
        while (!done_) {
          RemoveStraggler();
          sleep(1);
        }
      });
  }
  ~WorkloadPool() {
    done_ = true;
    if (straggler_killer_) {
      straggler_killer_->join();
      delete straggler_killer_;
    }
  }

  static void Match(const std::string& pattern, Workload* wl) {
    std::vector<std::string> files;
    MatchFile(pattern, &files);
    wl->file.resize(files.size());
    for (size_t i = 0; i < files.size(); ++i) {
      wl->file[i].filename = files[i];
    }
  }

  void Add(const std::vector<Workload::File>& files, int npart,
           const std::string& id = "") {
    std::lock_guard<std::mutex> lk(mu_);
    inited_ = true;
    for (auto f : files) {
      auto& t = task_[f.filename];
      if (t.track.empty()) t.track.resize(npart);
      CHECK_EQ(t.track.size(), npart);
      if (id.size()) t.node.insert(id);
    }
  }

  void Clear() {
    std::lock_guard<std::mutex> lk(mu_);
    task_.clear();
    assigned_.clear();
    inited_ = false;
    time_.clear();
    num_finished_ = 0;
  }

  void ClearRemain() {
    std::lock_guard<std::mutex> lk(mu_);
    task_.clear();
  }

  void Get(const std::string& id, Workload* wl) {
    std::lock_guard<std::mutex> lk(mu_);
    wl->file.clear();
    for (int i = 0; i < num_file_per_wl_; ++i) {
      GetOne(id, wl);
    }
  }

  // id dies
  void Reset(const std::string& id) { Set(id, false); }

  // id finished the workload it got before
  void Finish(const std::string& id) { Set(id, true); }

  bool IsFinished() {
    std::lock_guard<std::mutex> lk(mu_);
    return (inited_ && task_.empty() && assigned_.empty());
  }

  int num_finished() {
    std::lock_guard<std::mutex> lk(mu_);
    return num_finished_;
  }
  int num_assigned() {
    std::lock_guard<std::mutex> lk(mu_);
    return assigned_.size();
  }

 private:
  void Set(const std::string& id, bool del) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = assigned_.begin();
    while (it != assigned_.end()) {
      if (it->node == id) {
        if (!del) {
          Mark(it->filename, it->k, 0);
          LOG(INFO) << id << " failed to finish workload " << it->DebugStr();
        } else {
          double time = GetTime() - it->start;
          time_.push_back(time);
          Mark(it->filename, it->k, 2);
          LOG(INFO) << id << " finished " << it->DebugStr()
                    << " in " << time << " sec.";
        }
        it = assigned_.erase(it);
      } else {
        ++ it;
      }
    }
  }

  void GetOne(const std::string& id, Workload* wl) {
    for (auto& it : task_) {
      auto& t = it.second;
      if (!t.node.empty() && t.node.count(id) == 0) continue;
      if (t.done == t.track.size()) continue;
      for (auto& k : t.track) {
        if (k == 0) {
          Assigned a;
          a.filename = it.first;
          a.start    = GetTime();
          a.node     = id;
          a.k        = k;
          a.n        = (int)t.track.size();
          assigned_.push_back(a);
          wl->file.push_back(a.Get());
          LOG(INFO) << "assign " << id << " job " << a.DebugStr();
          k = 1;
          return;
        }
      }
    }
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
      double t = cur_t - it->start;
      if (t > std::max(mean * 2, (double)5)) {
        LOG(INFO) << it->node << " is processing "
                  << it->DebugStr() << " for " << t
                  << " sec, which is much longer than the average time "
                  << mean << " sec. reassign this workload to other nodes";
        Mark(it->filename, it->k, 0);
        it = assigned_.erase(it);
      } else {
        ++ it;
      }
    }
  }


  void Mark(const std::string& filename, int k, int mark) {
    auto it = task_.find(filename);
    if (it == task_.end()) return;
    auto& t = it->second;
    CHECK_LT(k, t.track.size());
    if (mark == 2 && t.track[k] != 2) {
      ++ num_finished_;
      ++ t.done;
    }
    t.track[k] = mark;
    if (t.done == t.track.size()) {
      task_.erase(it);
    }
  }

  int num_file_per_wl_ = 1;
  int num_finished_ = 0;

  struct Task {
    // capable nodes
    std::unordered_set<std::string> node;
    // 0: available, 1: assigned, 2: done
    std::vector<int> track;
    size_t done = 0;
  };
  std::unordered_map<std::string, Task> task_;


  struct Assigned {
    std::string filename;
    std::string node;
    int n, k;
    double start;  // start time
    Workload::File Get() {
      Workload::File f;
      f.filename = filename; f.n = n; f.k = k;
      return f;
    }
    std::string DebugStr() {
      std::stringstream ss;
      ss << filename << " " << k << "/" << n;
      return ss.str();
    }
  };

  std::list<Assigned> assigned_;

  bool inited_ = false, done_ = false;

  // process time of finished tasks
  std::vector<double> time_;
  std::mutex mu_;
  std::thread* straggler_killer_ = NULL;
};

}  // namespace dmlc

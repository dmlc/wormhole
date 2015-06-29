#pragma once
#include <sstream>
#include "dmlc/data.h"
#include "data/row_block.h"
namespace dmlc {

// debug string

template <typename V>
inline std::string DebugStr(const V* data, int n, int m = 5) {
  std::stringstream ss;
  ss << "[" << n << "]: ";
  if (n <= 2 * m) {
    for (int i = 0; i < n; ++i) ss << data[i] << " ";
  } else {
    for (int i = 0; i < m; ++i) ss << data[i] << " ";
    ss << "... ";
    for (int i = n-m; i < n; ++i) ss << data[i] << " ";
  }
  return ss.str();
}
template <typename V>
inline std::string DebugStr(const std::vector<V>& vec) {
  return DebugStr(vec.data(), vec.size());
}

template<typename I>
inline std::string DebugStr(const RowBlock<I>& blk) {
  std::stringstream ss;
  size_t idx_size = blk.offset[blk.size] - blk.offset[0];
  ss << "label: " << DebugStr<real_t>(blk.label, blk.size) << "\n"
     << "offset: " << DebugStr<size_t>(blk.offset, blk.size+1) << "\n"
     << "index: " << DebugStr<I>(blk.index, idx_size);
  if (blk.value) {
    ss << "\nvalue: " << DebugStr<real_t>(blk.value, idx_size);
  }
  return ss.str();
}

template<typename I>
inline std::string DebugStr(const data::RowBlockContainer<I>& blk) {
  return DebugStr(blk.GetBlock());
}


// for debug use
#define LL LOG(ERROR)

}  // namespace dmlc

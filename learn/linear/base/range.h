#pragma once
#include <dmlc/logging.h>
namespace dmlc {

/**
 * \brief a range between [begin, end)
 */
struct Range {
  Range(size_t _begin, size_t _end) : begin(_begin), end(_end) { }
  Range() : Range(0, 0) { }
  ~Range() { }

  /**
   * \brief evenly divide this range into npart segments, and return the idx-th
   * one
   */
  inline Range Segment(size_t idx, size_t nparts) const {
    CHECK_GE(end, begin);
    CHECK_GT(nparts, 0);
    CHECK_LT(idx, nparts);
    double itv = static_cast<double>(end - begin) /
                 static_cast<double>(nparts);
    size_t _begin = static_cast<size_t>(begin + itv * idx);
    size_t _end = (idx == nparts - 1) ?
                  end : static_cast<size_t>(begin + itv * (idx+1));
    return Range(_begin, _end);
  }

  /**
   * \brief Return true if i contains in this range
   */

  inline bool Has(size_t i) const {
    return (begin <= i && i < end);
  }

  size_t begin;
  size_t end;
};
}  // namespace dmlc

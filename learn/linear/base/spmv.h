#pragma once
#include <cstring>
#include "base/range.h"
#include "dmlc/data.h"
#include "dmlc/omp.h"
namespace dmlc {

/**
 * \brief multi-thread sparse matrix vector multiplication
 */
class SpMV {
 public:
  static const int kDefaultNT = 2;
  using SpMat = RowBlock<unsigned>;


  /** \brief y = D * x */
  template<typename V>
  static void Times(const SpMat& D, const std::vector<V>& x,
                    std::vector<V>* y, int nthreads = kDefaultNT) {
    CHECK_NOTNULL(y);
    CHECK_EQ(y->size(), D.size);
    Times<V>(D, x.data(), y->data(), nthreads);
  }

  /** \brief y = D^T * x */
  template<typename V>
  static void TransTimes(const SpMat& D, const std::vector<V>& x,
                    std::vector<V>* y, int nthreads = kDefaultNT) {
    CHECK_EQ(x.size(), D.size);
    CHECK_NOTNULL(y);
    TransTimes<V>(D, x.data(), y->data(), y->size(), nthreads);
  }

  /** \brief y = D * x */
  template<typename V>
  static void Times(const SpMat& D,  const V* const x, V* y, int nthreads = kDefaultNT) {
#pragma omp parallel num_threads(nthreads)
    {
      Range rg = Range(0, D.size).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = rg.begin; i < rg.end; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V y_i = 0;
        if (D.value) {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j)
            y_i += x[D.index[j]] * D.value[j];
        } else {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j)
            y_i += x[D.index[j]];
        }
        y[i] = y_i;
      }
    }
  }

  /** \brief y = D^T * x */
  template<typename V>
  static void TransTimes(const SpMat& D,  const V* const x, V* y, size_t y_size,
                         int nthreads = kDefaultNT) {
#pragma omp parallel num_threads(nthreads)
    {
      Range rg = Range(0, y_size).Segment(
          omp_get_thread_num(), omp_get_num_threads());
      std::memset(y + rg.begin, 0, sizeof(V) * (rg.end - rg.begin));

      for (size_t i = 0; i < D.size; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V x_i = x[i];
        if (D.value) {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            unsigned k = D.index[j];
            if (rg.Has(k)) y[k] += x_i * D.value[j];
          }
        } else {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            unsigned k = D.index[j];
            if (rg.Has(k)) y[k] += x_i;
          }
        }
      }
    }
  }

};
} // namespace dmlc

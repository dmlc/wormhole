/**
 * @file   spmm.h
 * @brief  sparse matrix dense matrix multiplication
 */
#pragma once
#include <cstring>
#include "dmlc/data.h"
#include "dmlc/omp.h"
#include "base/spmv.h"  // for Range

namespace dmlc {

/**
 * \brief multi-thread sparse matrix dense matrix multiplication
 */
class SpMM {
 public:
  static const int kDefaultNT = 2;
  using SpMat = RowBlock<unsigned>;

  /** \brief y = D * x */
  template<typename V>
  static void Times(const SpMat& D, const std::vector<V>& x,
                    std::vector<V>* y, int nt = kDefaultNT) {
    if (x.empty()) return;
    CHECK_NOTNULL(y);
    int dim = (int)(y->size() / D.size);
    Times<V>(D, x.data(), y->data(), dim, nt);
  }


  /** \brief y = D^T * x */
  template<typename V>
  static void TransTimes(const SpMat& D, const std::vector<V>& x,
                         std::vector<V>* y, int nt = kDefaultNT) {
    TransTimes<V>(D, x, 0, std::vector<V>(), y, nt);
  }

  /** \brief y = D^T * x + p * z */

  template<typename V>
  static void TransTimes(const SpMat& D, const std::vector<V>& x,
                         V p, const std::vector<V>& z,
                         std::vector<V>* y, int nt = kDefaultNT) {
    if (x.empty()) return;
    int dim = (int)(x.size() / D.size);
    if (z.size() == y->size() && p != 0) {
      TransTimes<V>(D, x.data(), z.data(), p, y->data(), y->size(), dim, nt);
    } else {
      TransTimes<V>(D, x.data(), NULL, 0, y->data(), y->size(), dim, nt);
    }
  }
 private:
  // y = D * x
  template<typename V>
  static void Times(const SpMat& D, const V* const x,
                    V* y, int dim, int nt = kDefaultNT) {
    memset(y, 0, D.size * dim * sizeof(V));
#pragma omp parallel num_threads(nt)
    {
      Range rg = Range(0, D.size).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = rg.begin; i < rg.end; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V* y_i = y + i * dim;
        if (D.value) {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            V const* x_j = x + D.index[j] * dim;
            V v = D.value[j];
            for (int k = 0; k < dim; ++k) y_i[k] += x_j[k] * v;
          }
        } else {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            V const* x_j = x + D.index[j] * dim;
            for (int k = 0; k < dim; ++k) y_i[k] += x_j[k];
          }
        }
      }
    }
  }

  // y = D' * x
  template<typename V>
  static void TransTimes(const SpMat& D, const V* const x,
                         const V* const z, V p,
                         V* y, size_t y_size, int dim,
                         int nt = kDefaultNT) {
    if (z) {
      for (size_t i = 0; i < y_size; ++i) y[i] = z[i] * p;
    } else {
      memset(y, 0, y_size*sizeof(V));
    }

#pragma omp parallel num_threads(nt)
    {
      Range rg = Range(0, y_size/dim).Segment(
          omp_get_thread_num(), omp_get_num_threads());

      for (size_t i = 0; i < D.size; ++i) {
        if (D.offset[i] == D.offset[i+1]) continue;
        V const * x_i = x + i * dim;
        if (D.value) {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            unsigned e = D.index[j];
            if (rg.Has(e)) {
              V v = D.value[j];
              V* y_j = y + e * dim;
              for (int k = 0; k < dim; ++k) y_j[k] += x_i[k] * v;
            }
          }
        } else {
          for (size_t j = D.offset[i]; j < D.offset[i+1]; ++j) {
            unsigned e = D.index[j];
            if (rg.Has(e)) {
              V* y_j = y + e * dim;
              for (int k = 0; k < dim; ++k) y_j[k] += x_i[k];
            }
          }
        }
      }
    }
  }
};

}  // namespace dmlc

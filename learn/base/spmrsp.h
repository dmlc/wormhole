#pragma once
#include <cstring>
#include "dmlc/data.h"
#include "dmlc/omp.h"
namespace dmlc {

/**
 * \brief row sparse matrix
 *
 * store each row in a dense vector but skip the rows with
 * all 0. For example, row 0 and 2 are skipped for the following matrix
 * [ 0  0  0
 *  .1 .2 .3
 *   0  0  0]
 */
template<typename V>
struct RowSpMat {
  void Init(V* value, int value_len, int cols) {
    CHECK_EQ(value_len % cols, 0);
    num_cols = cols;
    int num_rows = value_len / cols;
    row.resize(num_rows);
    for (int i = 0; i < num_rows, ++i) {
      row[i] = value + cols * i;
    }
  }

  void Init(V* value, int value_len, int* dim, int dim_len, int cols) {
    row.resize(dim_len);
    num_cols = cols;
    size_t pos = 0;
    for (int i = 0; i < dim_len; ++i) {
      row[i] = dim[i] == cols ? value + pos : NULL;
      pos += dim[i];
      CHECK_LE(pos, value_len);
    }
    CHECK_EQ(pos, value_len);
  }

  // number of  columns
  int num_cols = 0;
  // the pointer to the begin of row[i]. NULL if all 0
  std::vector<V*> row;
};


/**
 * \brief multi-thread sparse matrix times row sparse matrix
 */
class SpMRSp {
 public:
  static const int kDefaultNT = 2;
  using SpMat = RowBlock<unsigned>;

  /** \brief y = D * x */
  template<typename V>
  static void Times(const SpMat& D, const RowSpMat<V>& x,
                    RowSpMat<V>* y, int nthreads = kDefaultNT) {
  }

  /** \brief y = D^T * x */
  template<typename V>
  static void TransTimes(const SpMat& D, const RowSpMat<V>& x,
                    RowSpMat<V>* y, int nthreads = kDefaultNT) {
  }
};

} // namespace dmlc

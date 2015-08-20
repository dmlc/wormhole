#pragma once
#include <type_traits>
#include <limits>
#include "dmlc/data.h"
#include "dmlc/omp.h"
#include "data/row_block.h"
#include "base/parallel_sort.h"

namespace ps {
DECLARE_uint64(max_key);
}  // namespace ps

namespace dmlc {

/// \brief reverse the bytes of x to make it more uniformly spanning the space
inline uint64_t ReverseBytes(uint64_t x) {
  // return x;
  x = x << 32 | x >> 32;
  x = (x & 0x0000FFFF0000FFFFULL) << 16 |
      (x & 0xFFFF0000FFFF0000ULL) >> 16;
  x = (x & 0x00FF00FF00FF00FFULL) << 8 |
      (x & 0xFF00FF00FF00FF00ULL) >> 8;
  x = (x & 0x0F0F0F0F0F0F0F0FULL) << 4 |
      (x & 0xF0F0F0F0F0F0F0F0ULL) >> 4;
  return x;
}

/**
 * @brief Mapping a RowBlock with general indices into continuous indices
 * starting from 0
 * @tparam I the index type
 */
template<typename I>
class Localizer {
 public:
  Localizer(int nthreads = 2) : nt_(nthreads) { }
  ~Localizer() { }
  /**
   * @brief Localize a Rowblock
   */
  template<typename C = unsigned>
  void Localize(const RowBlock<I>& blk,
                data::RowBlockContainer<unsigned> *localized,
                std::vector<I>* uniq_idx = NULL,
                std::vector<C>* idx_frq = NULL) {
    std::vector<I>* uidx = uniq_idx == NULL ? new std::vector<I>() : uniq_idx;
    CountUniqIndex<C>(blk, uidx, idx_frq);
    RemapIndex(blk, *uidx, localized);
    if (uniq_idx == NULL) delete uidx;
    Clear();
  }

  /**
   * @brief count unique items
   *
   * temporal results will be stored to accelerate RemapIndex().
   *
   * @param idx the item list in any order
   * @param uniq_idx returns the sorted unique items
   * @param idx_frq if not NULL then returns the according occurrence counts
   */
  template<typename C>
  void CountUniqIndex(const RowBlock<I>& blk,
                      std::vector<I> *uniq_idx,
                      std::vector<C>* idx_frq);

  /**
   * @brief Remaps the index.
   *
   * @param idx_dict the index dictionary. Any index does not exists in this
   * dictionary is dropped.
   *
   * @param localized a rowblock with index mapped: idx_dict[i] -> i.
   */
  void RemapIndex(const RowBlock<I>& blk,
                  const std::vector<I>& idx_dict,
                  data::RowBlockContainer<unsigned> *localized);


  /**
   * @brief Clears the temporal results
   */
  void Clear() { pair_.clear(); }

 private:
  int nt_;
#pragma pack(push)
#pragma pack(4)
  struct Pair {
    I k; unsigned i;
  };
#pragma pack(pop)
  std::vector<Pair> pair_;
};

template<typename I>
template<typename C>
void Localizer<I>:: CountUniqIndex(
    const RowBlock<I>& blk, std::vector<I> *uniq_idx, std::vector<C>* idx_frq) {
  // sort
  if (blk.size == 0) return;
  size_t idx_size = blk.offset[blk.size];
  CHECK_LT(idx_size, static_cast<size_t>(std::numeric_limits<unsigned>::max()))
      << "you need to change Pair.i from unsigned to uint64";
  pair_.resize(idx_size);

  I max_index = std::numeric_limits<I>::max();
  if (ps::FLAGS_max_key < (uint64_t)max_index) {
    // hash kernel
    max_index = (I) ps::FLAGS_max_key;
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < idx_size; ++i) {
      pair_[i].k = blk.index[i] % max_index;
      pair_[i].i = i;
    }
  } else if (sizeof(I) == 8) {
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < idx_size; ++i) {
      pair_[i].k = ReverseBytes(blk.index[i]);
      pair_[i].i = i;
    }
  } else {
#pragma omp parallel for num_threads(nt_)
    for (size_t i = 0; i < idx_size; ++i) {
      pair_[i].k = blk.index[i];
      pair_[i].i = i;
    }
  }

  ParallelSort(&pair_, nt_,
               [](const Pair& a, const Pair& b) {return a.k < b.k; });

  // save data
  CHECK_NOTNULL(uniq_idx);
  uniq_idx->clear();
  if (idx_frq) idx_frq->clear();

  // cnt_max doesn't work for float and double
  bool int_cnt = std::is_integral<C>::value;
  unsigned cnt_max = static_cast<unsigned>(std::numeric_limits<C>::max());
  I curr = pair_[0].k;
  unsigned cnt = 0;
  for (size_t i = 0; i < pair_.size(); ++i) {
    const Pair& v = pair_[i];
    if (v.k != curr) {
      uniq_idx->push_back(curr);
      curr = v.k;
      if (idx_frq) {
        if (int_cnt) {
          idx_frq->push_back(std::min(cnt, cnt_max));
        } else {
          idx_frq->push_back(static_cast<C>(cnt));
        }
      }
      cnt = 0;
    }
    ++ cnt;
  }
  uniq_idx->push_back(curr);
  if (idx_frq) {
    if (int_cnt) {
      idx_frq->push_back(std::min(cnt, cnt_max));
    } else {
      idx_frq->push_back(static_cast<C>(cnt));
    }
  }
}

template<typename I>
void Localizer<I>::RemapIndex(
    const RowBlock<I>& blk, const std::vector<I>& idx_dict,
    data::RowBlockContainer<unsigned> *localized) {
  if (blk.size == 0 || idx_dict.empty()) return;
  CHECK_LT(idx_dict.size(),
           static_cast<size_t>(std::numeric_limits<unsigned>::max()));
  CHECK_EQ(blk.offset[blk.size], pair_.size());

  // build the index mapping
  unsigned matched = 0;
  std::vector<unsigned> remapped_idx(pair_.size(), 0);
  auto cur_dict = idx_dict.cbegin();
  auto cur_pair = pair_.cbegin();
  while (cur_dict != idx_dict.cend() && cur_pair != pair_.cend()) {
    if (*cur_dict < cur_pair->k) {
      ++ cur_dict;
    } else {
      if (*cur_dict == cur_pair->k) {
        remapped_idx[cur_pair->i]
            = static_cast<unsigned>((cur_dict-idx_dict.cbegin()) + 1);
        ++ matched;
      }
      ++ cur_pair;
    }
  }

  // construct the new rowblock
  data::RowBlockContainer<unsigned>* o = localized;
  CHECK_NOTNULL(o);
  o->offset.resize(blk.size+1); o->offset[0] = 0;
  o->index.resize(matched);
  if (blk.value) o->value.resize(matched);

  size_t k = 0;
  for (size_t i = 0; i < blk.size; ++i) {
    size_t n = 0;
    for (size_t j = blk.offset[i]; j < blk.offset[i+1]; ++j) {
      if (remapped_idx[j] == 0) continue;
      ++ n;
      if (blk.value) o->value[k] = blk.value[j];
      o->index[k++] = remapped_idx[j] - 1;
    }
    o->offset[i+1] = o->offset[i] + n;
  }
  CHECK_EQ(k, matched);

  if (blk.label) {
    o->label.resize(blk.size);
    memcpy(o->label.data(), blk.label, blk.size*sizeof(real_t));
  }
  o->max_index = idx_dict.size() - 1;
}

}  // namespace dmlc

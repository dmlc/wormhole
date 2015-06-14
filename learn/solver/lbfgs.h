/*!
 *  Copyright (c) 2015 by Contributors
 * \file lbfgs.h
 * \brief L-BFGS solver for general optimization problem
 *
 * \author Tianqi Chen
 */
#ifndef DMLC_LEARN_LBFGS_H_
#define DMLC_LEARN_LBFGS_H_
#include <cmath>
#include <rabit.h>
#include <dmlc/logging.h>

namespace dmlc {
/*! \brief namespace of solver for general problems */
namespace solver {
/*!
 * \brief objective function for optimizers 
 *  the objective function can also implement save/load
 *  to remember the state parameters that might need to remember
 */
template<typename DType>
class IObjFunction : public rabit::Serializable {
 public:
  // destructor
  virtual ~IObjFunction(void){}
  /*!
   * \brief evaluate function values for a given weight
   * \param weight weight of the function
   * \param size size of the weight
   */
  virtual double Eval(const DType *weight, size_t size) = 0;
  /*!
   * \return number of feature dimension to be allocated
   * only called once during initialization
   */
  virtual size_t InitNumDim(void) = 0;
  /*!
   * \brief initialize the weight before starting the solver
   * only called once for initialization
   */
  virtual void InitModel(DType *weight, size_t size) = 0;
  /*!
   * \brief calculate gradient for a given weight
   * \param out_grad used to store the gradient value of the function
   * \param weight weight of the function
   * \param size size of the weight
   */
  virtual void CalcGrad(DType *out_grad,
                        const DType *weight,
                        size_t size) = 0;
};

/*! \brief a basic version L-BFGS solver */
template<typename DType>
class LBFGSSolver {
 public:
  LBFGSSolver(void) {
    // set default values
    reg_L1 = 0.0f;
    max_linesearch_iter = 100;
    linesearch_backoff = 0.5f;
    linesearch_c1 = 1e-4;
    min_lbfgs_iter = 5;
    max_lbfgs_iter = 500;
    lbfgs_stop_tol = 1e-5f;
    silent = 0;
  }
  virtual ~LBFGSSolver(void) {}
  /*!
   * \brief set parameters from outside
   * \param name name of the parameter
   * \param val value of the parameter
   */
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp("num_dim", name)) {
      gstate.num_dim = static_cast<size_t>(atol(val));
    }
    if (!strcmp("size_memory", name)) {
      gstate.size_memory = static_cast<size_t>(atol(val));
    }
    if (!strcmp("reg_L1", name)) {
      reg_L1 = static_cast<float>(atof(val));
    }
    if (!strcmp("lbfgs_stop_tol", name)) {
      lbfgs_stop_tol = static_cast<float>(atof(val));
    }
    if (!strcmp("linesearch_backoff", name)) {
      linesearch_backoff = static_cast<float>(atof(val));
    }
    if (!strcmp("max_linesearch_iter", name)) {
      max_linesearch_iter = atoi(val);
    }
    if (!strcmp("max_lbfgs_iter", name)) {
      max_lbfgs_iter = atoi(val);
    }
    if (!strcmp("min_lbfgs_iter", name)) {
      min_lbfgs_iter = atoi(val);
    }
    if (!strcmp("linesearch_c1", name)) {
      linesearch_c1 = static_cast<float>(atof(val));
    }
  }
  /*!
   * \brief set objective function to optimize
   *  the objective function only need to evaluate and calculate
   *  gradient with respect to current subset of data
   * \param obj the objective function we are looking for
   */
  virtual void SetObjFunction(IObjFunction<DType> *obj) {
    gstate.obj = obj;
  }
  /*!
   * \brief initialize the LBFGS solver
   *  user must already set the objective function
   */
  virtual void Init(void) {
    CHECK(gstate.obj != NULL) <<
        "LBFGSSolver.Init must SetObjFunction first";
    int version = rabit::LoadCheckPoint(&gstate, &hist);
    if (version == 0) {
      gstate.num_dim = gstate.obj->InitNumDim();
    } else {
      printf("restart from version=%d\n", version);
    }
    {
      // decide parameter partition
      size_t nproc = rabit::GetWorldSize();
      size_t rank = rabit::GetRank();
      size_t step = (gstate.num_dim + nproc - 1) / nproc;
      // upper align
      step = (step + 7) / 8 * 8;
      CHECK(step * nproc >= gstate.num_dim);
      range_begin_ = std::min(rank * step, gstate.num_dim);
      range_end_ = std::min((rank + 1) * step, gstate.num_dim);
    }
    if (version == 0) {
      gstate.Init();
      hist.Init(range_end_ - range_begin_, gstate.size_memory);
      gstate.obj->InitModel(gstate.weight, gstate.num_dim);
      // broadcast initialize model
      rabit::Broadcast(gstate.weight,
                       sizeof(DType) * gstate.num_dim, 0);
      gstate.old_objval = this->Eval(gstate.weight);
      gstate.init_objval = gstate.old_objval;
      
      if (silent == 0 && rabit::GetRank() == 0) {
        rabit::TrackerPrintf
            ("L-BFGS solver starts, num_dim=%lu, init_objval=%g, size_memory=%lu, RAM-approx=%lu\n",
             gstate.num_dim, gstate.init_objval, gstate.size_memory,
             gstate.MemCost() + hist.MemCost());
      }
    }
  }
  /*!
   * \brief get the current weight vector
   *  note that if update function is called
   *  the content of weight vector is no longer valid
   * \return weight vector
   */
  virtual DType *GetWeight(void) {
    return gstate.weight;
  }
  /*!
   * \brief update the weight for one L-BFGS iteration
   * \return whether stopping condition is met
   */
  virtual bool UpdateOneIter(void) {
    bool stop = false;
    GlobalState &g = gstate;
    g.obj->CalcGrad(g.grad, g.weight, g.num_dim);
    rabit::Allreduce<rabit::op::Sum>(g.grad, g.num_dim);
    // find change direction
    double vdot = FindChangeDirection(g.tempw, g.grad, g.weight);
    // line-search, g.grad is now new weight
    int iter = BacktrackLineSearch(g.grad, g.tempw, g.weight, vdot);
    CHECK(iter < max_linesearch_iter) << "line search failed";
    // swap new weight 
    std::swap(g.weight, g.grad);
    // check stop condition
    if (gstate.num_iteration > static_cast<size_t>(min_lbfgs_iter)) {
      if (g.old_objval - g.new_objval < lbfgs_stop_tol * g.init_objval) {
        return true;
      }
    }
    if (silent == 0 && rabit::GetRank() == 0) {
      rabit::TrackerPrintf
          ("[%d] L-BFGS: linesearch finishes in %d rounds, new_objval=%g, improvment=%g\n",
           gstate.num_iteration, iter,
           gstate.new_objval,
           gstate.old_objval - gstate.new_objval);
    }
    gstate.old_objval = gstate.new_objval;
    rabit::CheckPoint(&gstate, &hist);
    return stop;
  }
  /*! \brief run optimization */
  virtual void Run(void) {
    this->Init();
    while (gstate.num_iteration < static_cast<size_t>(max_lbfgs_iter)) {
      if (this->UpdateOneIter()) break;
    }
    if (silent == 0 && rabit::GetRank() == 0) {
      size_t nonzero = 0;
      for (size_t i = 0; i < gstate.num_dim; ++i) {
        if (gstate.weight[i] != 0.0f) nonzero += 1;
      }
      rabit::TrackerPrintf
          ("L-BFGS: finishes at iteration %d, %lu/%lu active weights\n",
           gstate.num_iteration, nonzero, gstate.num_dim);
    }
  }
 protected:
  // find the delta value, given gradient
  // return dot(dir, l1grad)
  virtual double FindChangeDirection(DType *dir,
                                     const DType *grad,
                                     const DType *weight) {
    int m = static_cast<int>(gstate.size_memory);
    int n = static_cast<int>(hist.num_useful());
    if (n < m) {
      CHECK(hist.num_useful() == gstate.num_iteration);
    } else {
      CHECK(n == m);
    }
    const size_t num_dim = gstate.num_dim;
    const DType *gsub = grad + range_begin_;
    const size_t nsub = range_end_ - range_begin_;
    double vdot = 0.0;
    if (n != 0) {
      // hist[m + n - 1] stores old gradient
      Minus(hist[m + n - 1], gsub, hist[m + n - 1], nsub);
      SetL1Dir(hist[2 * m], gsub, weight + range_begin_, nsub);
      // index set for calculating results
      std::vector<std::pair<size_t, size_t> > idxset;
      for (int j = 0; j < n; ++j) {
        idxset.push_back(std::make_pair(j, 2 * m));
        idxset.push_back(std::make_pair(j, n - 1));
        idxset.push_back(std::make_pair(j, m + n - 1));
      }
      for (int j = 0; j < n; ++j) {
        idxset.push_back(std::make_pair(m + j, 2 * m));
        idxset.push_back(std::make_pair(m + j, m + n - 1));
      }

      // calculate dot products
      std::vector<double> tmp(idxset.size());
      for (size_t i = 0; i < tmp.size(); ++i) {
        tmp[i] = hist.CalcDot(idxset[i].first, idxset[i].second);
      }

      rabit::Allreduce<rabit::op::Sum>(BeginPtr(tmp), tmp.size());

      for (size_t i = 0; i < tmp.size(); ++i) {
        gstate.DotBuf(idxset[i].first, idxset[i].second) = tmp[i];
      }

      // BFGS steps, use vector-free update
      // parameterize vector using basis in hist
      std::vector<double> alpha(n);
      std::vector<double> delta(2 * m + 1, 0.0);
      delta[2 * m] = 1.0;
      // backward step
      for (int j = n - 1; j >= 0; --j) {
        double vsum = 0.0;
        for (size_t k = 0; k < delta.size(); ++k) {
          vsum += delta[k] * gstate.DotBuf(k, j);
        }
        alpha[j] = vsum / gstate.DotBuf(j, m + j);
        delta[m + j] = delta[m + j] - alpha[j];
      }      
      // scale
      double scale = gstate.DotBuf(n - 1, m + n - 1) /
      gstate.DotBuf(m + n - 1, m + n - 1);
      for (size_t k = 0; k < delta.size(); ++k) {
        delta[k] *= scale;
      }
      // forward step
      for (int j = 0; j < n; ++j) {
        double vsum = 0.0;
        for (size_t k = 0; k < delta.size(); ++k) {
          vsum += delta[k] * gstate.DotBuf(k, m + j);
        }
        double beta = vsum / gstate.DotBuf(j, m + j);
        delta[j] = delta[j] + (alpha[j] - beta);
      }

      // set all to zero
      std::fill(dir, dir + num_dim, 0.0f);
      DType *dirsub = dir + range_begin_; 
      for (int i = 0; i < n; ++i) {
        AddScale(dirsub, dirsub, hist[m + i], delta[m + i], nsub);
      }
      AddScale(dirsub, dirsub, hist[2 * m], delta[2 * m], nsub);
      for (int i = 0; i < n; ++i) {
        AddScale(dirsub, dirsub, hist[i], delta[i], nsub);
      }
      FixDirL1Sign(dirsub, hist[2 * m], nsub);
      vdot = -Dot(dirsub, hist[2 * m], nsub);

      // allreduce to get full direction
      rabit::Allreduce<rabit::op::Sum>(dir, num_dim);
      rabit::Allreduce<rabit::op::Sum>(&vdot, 1);
    } else {
      SetL1Dir(dir, grad, weight, num_dim);
      vdot = -Dot(dir, dir, num_dim);
    }
    // shift the history record    
    if (n < m) {
      n += 1;
    } else {
      gstate.Shift(); hist.Shift();
    }
    hist.set_num_useful(n);
    // copy gradient to hist[m + n - 1]
    memcpy(hist[m + n - 1], gsub, nsub * sizeof(DType));
    return vdot;
  }
  // line search for given direction
  // return whether there is a descent
  inline int BacktrackLineSearch(DType *new_weight,
                                 const DType *dir,
                                 const DType *weight,
                                 double dot_dir_l1grad) {
    CHECK(dot_dir_l1grad < 0.0f);
    double alpha = 1.0;
    double backoff = linesearch_backoff;
    // unit descent direction in first iter
    if (gstate.num_iteration == 0) {
      CHECK(hist.num_useful() == 1);
      alpha = 1.0f / std::sqrt(-dot_dir_l1grad);
      backoff = 0.1f;
    }
    int iter = 0;
    
    double old_val = gstate.old_objval;
    double c1 = this->linesearch_c1;
    while (true) {
      const size_t num_dim = gstate.num_dim;
      if (++iter >= max_linesearch_iter) return iter;
      AddScale(new_weight, weight, dir, alpha, num_dim);
      this->FixWeightL1Sign(new_weight, weight, num_dim);
      double new_val = this->Eval(new_weight);
      if (new_val - old_val <= c1 * dot_dir_l1grad * alpha) {
        gstate.new_objval = new_val; break;
      }
      alpha *= backoff;
    }
    // hist[n - 1] = new_weight - weight
    Minus(hist[hist.num_useful() - 1],
          new_weight + range_begin_,
          weight + range_begin_,
          range_end_ - range_begin_);
    gstate.num_iteration += 1;
    return iter;
  }
  // OWL-QN step for L1 regularization
  inline void SetL1Dir(DType *dst,
                       const DType *grad,
                       const DType *weight,
                       size_t size) {
    if (reg_L1 == 0.0) {
      for (size_t i = 0; i < size; ++i) {
        dst[i] = -grad[i];
      }
    } else {
      for (size_t i = 0; i < size; ++i) {
        if (weight[i] > 0.0f) {
          dst[i] = -grad[i] - reg_L1;
        } else if (weight[i] < 0.0f) {
          dst[i] = -grad[i] + reg_L1;
        } else {
          if (grad[i] < -reg_L1) {
            dst[i] = -grad[i] - reg_L1;
          } else if (grad[i] > reg_L1) {
            dst[i] = -grad[i] + reg_L1;
          } else {
            dst[i] = 0.0;
          }
        }
      }
    }
  }
  // OWL-QN step: fix direction sign to be consistent with proposal
  inline void FixDirL1Sign(DType *dir,
                           const DType *steepdir,
                           size_t size) {
    if (reg_L1 != 0.0f) {
      for (size_t i = 0; i < size; ++i) {
        if (dir[i] * steepdir[i] <= 0.0f) {
          dir[i] = 0.0f;
        }
      }
    }
  }
  // QWL-QN step: fix direction sign to be consistent with proposal
  inline void FixWeightL1Sign(DType *new_weight,
                              const DType *weight,
                              size_t size) {
    if (reg_L1 != 0.0f) {
      for (size_t i = 0; i < size; ++i) {
        if (new_weight[i] * weight[i] < 0.0f) {
          new_weight[i] = 0.0f;
        }
      }
    }
  }
  inline double Eval(const DType *weight) {
    double val = gstate.obj->Eval(weight, gstate.num_dim);    
    rabit::Allreduce<rabit::op::Sum>(&val, 1);
    if (reg_L1 != 0.0f) {
      double l1norm = 0.0;
      for (size_t i = 0; i < gstate.num_dim; ++i) {
        l1norm += std::abs(weight[i]);
      }
      val += l1norm * reg_L1;
    }
    return val;
  }

 private:
  // helper functions
  // dst = lhs + rhs * scale
  inline static void AddScale(DType *dst,
                              const DType *lhs,
                              const DType *rhs,
                              DType scale,
                              size_t size) {
    for (size_t i = 0; i < size; ++i) {
      dst[i] = lhs[i] + rhs[i] * scale;
    }
  }
  // dst = lhs - rhs
  inline static void Minus(DType *dst,
                           const DType *lhs,
                           const DType *rhs,
                           size_t size) {
    for (size_t i = 0; i < size; ++i) {
      dst[i] = lhs[i] - rhs[i];
    }
  }
  // return dot(lhs, rhs)
  inline static double Dot(const DType *lhs,
                           const DType *rhs,
                           size_t size) {
    double res = 0.0;
    for (size_t i = 0; i < size; ++i) {
      res += lhs[i] * rhs[i];
    }
    return res;
  }
  // map rolling array index
  inline static size_t MapIndex(size_t i, size_t offset,
                                size_t size_memory) {
    if (i == 2 * size_memory) return i;
    if (i < size_memory) {
      return (i + offset) % size_memory;
    } else {
      CHECK(i < 2 * size_memory);
      return (i + offset) % size_memory + size_memory;
    }
  }
  // global solver state
  struct GlobalState : public rabit::Serializable {
   public:
    // memory size of L-BFGS
    size_t size_memory;
    // number of iterations passed
    size_t num_iteration;
    // number of features in the solver
    size_t num_dim;
    // initialize objective value
    double init_objval;
    // history objective value
    double old_objval;
    // new objective value
    double new_objval;
    // objective function
    IObjFunction<DType> *obj;
    // temporal storage
    DType *grad, *weight, *tempw;
    // constructor
    GlobalState(void)
        : obj(NULL), grad(NULL),
          weight(NULL), tempw(NULL) {
      size_memory = 10;
      num_iteration = 0;
      num_dim = 0;
      old_objval = 0.0;
      offset_ = 0;
    }
    ~GlobalState(void) {
      if (grad != NULL) {
        delete [] grad;
        delete [] weight;
        delete [] tempw;
      }
    }
    // intilize the space of rolling array
    inline void Init(void) {
      size_t n = size_memory * 2 + 1;
      data.resize(n * n, 0.0);
      this->AllocSpace();
    }
    // memory cost
    inline size_t MemCost(void) const {
      return sizeof(DType) * 3 * num_dim;
    }
    inline double &DotBuf(size_t i, size_t j)  {
      if (i > j) std::swap(i, j);
      return data[MapIndex(i, offset_, size_memory) * (size_memory * 2 + 1) +
                  MapIndex(j, offset_, size_memory)];
    }
    // load the shift array
    virtual void Load(rabit::Stream *fi) {
      fi->Read(&size_memory, sizeof(size_memory));
      fi->Read(&num_iteration, sizeof(num_iteration));
      fi->Read(&num_dim, sizeof(num_dim));
      fi->Read(&init_objval, sizeof(init_objval));
      fi->Read(&old_objval, sizeof(old_objval));
      fi->Read(&offset_, sizeof(offset_));
      fi->Read(&data);
      this->AllocSpace();
      fi->Read(weight, sizeof(DType) * num_dim);
      obj->Load(fi);
    }
    // save the shift array
    virtual void Save(rabit::Stream *fo) const {
      fo->Write(&size_memory, sizeof(size_memory));
      fo->Write(&num_iteration, sizeof(num_iteration));
      fo->Write(&num_dim, sizeof(num_dim));
      fo->Write(&init_objval, sizeof(init_objval));
      fo->Write(&old_objval, sizeof(old_objval));
      fo->Write(&offset_, sizeof(offset_));
      fo->Write(data);
      fo->Write(weight, sizeof(DType) * num_dim);
      obj->Save(fo);
    }
    inline void Shift(void) {
      offset_ = (offset_ + 1) % size_memory;
    }
    
   private:    
    // rolling offset in the current memory
    size_t offset_;
    std::vector<double> data;
    // allocate sapce
    inline void AllocSpace(void) {
      if (grad == NULL) {
        grad = new DType[num_dim];
        weight = new DType[num_dim];
        tempw = new DType[num_dim];
      }
    }
  };
  /*! \brief rolling array that carries history information */
  struct HistoryArray : public rabit::Serializable {
   public:
    HistoryArray(void) : dptr_(NULL) {
      num_useful_ = 0;
    }
    ~HistoryArray(void) {
      if (dptr_ != NULL) delete [] dptr_;
    }
    // intilize the space of rolling array
    inline void Init(size_t num_col, size_t size_memory) {
      if (dptr_ != NULL &&
          (num_col_ != num_col || size_memory_ != size_memory)) {
        delete dptr_;
      }
      num_col_ = num_col;
      size_memory_ = size_memory;
      stride_ = num_col_;
      offset_ = 0;
      size_t n = size_memory * 2 + 1;
      dptr_ = new DType[n * stride_];
    }
    // memory cost
    inline size_t MemCost(void) const {
      return sizeof(DType) * (size_memory_ * 2 + 1) * stride_;
    }
    // fetch element from rolling array
    inline const DType *operator[](size_t i) const {
      return dptr_ + MapIndex(i, offset_, size_memory_) * stride_;
    }
    inline DType *operator[](size_t i) {
      return dptr_ + MapIndex(i, offset_, size_memory_) * stride_;
    }
    // shift array: arr_old -> arr_new
    // for i in [0, size_memory - 1), arr_new[i] = arr_old[i + 1]
    // for i in [size_memory, 2 * size_memory - 1), arr_new[i] = arr_old[i + 1]
    // arr_old[0] and arr_arr[size_memory] will be discarded
    inline void Shift(void) {
      offset_ = (offset_ + 1) % size_memory_;
    }
    inline double CalcDot(size_t i, size_t j) const {
      return Dot((*this)[i], (*this)[j], num_col_);
    }
    // set number of useful memory
    inline const size_t &num_useful(void) const {
      return num_useful_;
    }
    // set number of useful memory
    inline void set_num_useful(size_t num_useful) {
      CHECK(num_useful <= size_memory_);
      num_useful_ = num_useful;
    }
    // load the shift array
    virtual void Load(rabit::Stream *fi) {
      fi->Read(&num_col_, sizeof(num_col_));
      fi->Read(&stride_, sizeof(stride_));
      fi->Read(&size_memory_, sizeof(size_memory_));
      fi->Read(&num_useful_, sizeof(num_useful_));
      this->Init(num_col_, size_memory_);
      for (size_t i = 0; i < num_useful_; ++i) {
        fi->Read((*this)[i], num_col_ * sizeof(DType));
        fi->Read((*this)[i + size_memory_], num_col_ * sizeof(DType));
      }
    }
    // save the shift array
    virtual void Save(rabit::Stream *fo) const {
      fo->Write(&num_col_, sizeof(num_col_));
      fo->Write(&stride_, sizeof(stride_));
      fo->Write(&size_memory_, sizeof(size_memory_));
      fo->Write(&num_useful_, sizeof(num_useful_));
      for (size_t i = 0; i < num_useful_; ++i) {
        fo->Write((*this)[i], num_col_ * sizeof(DType));
        fo->Write((*this)[i + size_memory_], num_col_ * sizeof(DType));
      }
    }

   private:
    // number of columns in each of array
    size_t num_col_;
    // stride for each of column for alignment
    size_t stride_;
    // memory size of L-BFGS
    size_t size_memory_;
    // number of useful memory that will be used
    size_t num_useful_;
    // rolling offset in the current memory
    size_t offset_;
    // data pointer
    DType *dptr_;
  };
  // data structure for LBFGS
  GlobalState gstate;
  HistoryArray hist;
  // silent
  int silent;
  // the subrange of current node
  size_t range_begin_;
  size_t range_end_;
  // L1 regularization co-efficient
  float reg_L1;
  // c1 ratio for line search
  float linesearch_c1;
  float linesearch_backoff;
  int max_linesearch_iter;
  int max_lbfgs_iter;
  int min_lbfgs_iter;
  float lbfgs_stop_tol;
};
}  // namespace solver
}  // namespace dmlc
#endif // DMLC_LEARN_LBFGS_H_

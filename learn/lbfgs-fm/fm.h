#ifndef DMLC_LEARN_FM_H_
#define DMLC_LEARN_FM_H_
#include <omp.h>
#include <dmlc/data.h>
#include "lbfgs.h"

namespace dmlc {
namespace fm {
/*! \brief simple linear model */
struct FmModel {
  struct ModelParam {
    /*! \brief factor num */
    int nfactor;
    /*! \brief global bias */
    float base_score;
    /*! \brief number of samples */
    size_t num_size;
    /*! \brief number of validation samples */
    size_t num_size_val;
    /*! \brief number of features */
    size_t num_feature;
    /*! \brife number of weight */
    size_t num_weight;
    /*! \brief loss type*/
    int loss_type;
    // reserved field
    int reserved[16];
    // constructor
    ModelParam(void) {
      memset(this, 0, sizeof(ModelParam));
      nfactor = 4;
      base_score = 0.5f;
      num_feature = 0;
      num_weight = 0;
      num_size = 0;
      num_size_val = 0;
      loss_type = 1;
    }
    // initialize base score
    inline void InitBaseScore(void) {
      CHECK(base_score > 0.0f && base_score < 1.0f) <<
          "base_score must be in (0,1) for logistic loss";
      base_score = -std::log(1.0f / base_score - 1.0f);
    }
    /*!
     * \brief set parameters from outside
     * \param name name of the parameter
     * \param val value of the parameter
     */    
    inline void SetParam(const char *name, const char *val) {
      using namespace std;
      if (!strcmp("nfactor",name)) {
        nfactor = atoi(val);
      }
      if (!strcmp("base_score", name)) {
        base_score = static_cast<float>(atof(val));
      }
      if (!strcmp("num_feature", name)) {
        num_feature = static_cast<size_t>(atol(val));
      }
      if (!strcmp("objective", name)) {
        if (!strcmp("linear", val)) {
          loss_type = 0;
        } else if (!strcmp("logistic", val)) {
          loss_type = 1;
        } else {
          LOG(FATAL) << "unknown objective type " << val;
        }
      }
    }
    // transform margin to prediction
    inline float MarginToPred(float margin) const {
      if (loss_type == 1) {
        return 1.0f / (1.0f + std::exp(-margin));
      } else {
        return margin;
      }
    }
    // margin to loss
    inline float MarginToLoss(float label, float margin) const {
      if (loss_type == 1) {
        float nlogprob;
        if (margin > 0.0f) {
          nlogprob = std::log(1.0f + std::exp(-margin));
        } else {
          nlogprob = -margin + std::log(1.0f + std::exp(margin));
        }
        return label * nlogprob +
            (1.0f -label) * (margin + nlogprob); 
      } else {
        float diff = margin - label;
        return 0.5f * diff * diff;
      }
    }
    inline float PredToGrad(float label, float pred) const {
      return pred - label;      
    }
    inline float PredictMargin(const float *weight,
                               const Row<unsigned> &v) const {
      // weight[num_weight-1] is bias
      float sum = base_score + weight[num_weight - 1];
      // single features X*w
      for (unsigned i = 0; i < v.length; ++i) {
        if (v.index[i] >= num_weight) continue;
        sum += weight[v.index[i]] * v.get_value(i);
      }
      // pairwisie interation features 0.5 * sum((X*V).^2 - (X.*X)*(V.*V), 2)
      float sum1, sum2;
      for (int i = 0; i < nfactor; ++i) {
        sum1 = 0.f;
        sum2 = 0.f;
        for (unsigned j = 0; j < v.length; ++j) {
          size_t n = num_feature  + v.index[j] * nfactor + i;
          if (n >= num_weight) continue;
          float XV = weight[n] * v.get_value(j);
          sum1 += XV;
          sum2 += XV * XV;
        }
        sum += 0.5 * (sum1 * sum1 - sum2);
      }
      return sum;
    }
    inline float Predict(const float *weight,
                         const Row<unsigned> &v) const {
      return MarginToPred(PredictMargin(weight, v));
    }
  };
  // model parameter
  ModelParam param;
  // weight corresponding to the model
  float *weight;
  FmModel(void) : weight(NULL) {
  }
  ~FmModel(void) {
    if (weight != NULL) delete [] weight;
  }
  // load model
  inline void Load(dmlc::Stream *fi) {
    fi->Read(&param, sizeof(param));
    if (weight == NULL) {
      weight = new float[param.num_weight];
    }
    fi->Read(weight, sizeof(float) * (param.num_weight));
  }
  inline void Save(dmlc::Stream *fo, const float *wptr = NULL) {
    fo->Write(&param, sizeof(param));
    if (wptr == NULL) wptr = weight;
    fo->Write(wptr, sizeof(float) * (param.num_weight));
  }
  inline float Predict(const Row<unsigned> &v) const {
    return param.Predict(weight, v);
  }
};
}  // namespace fm
}  // namespace dmlc
#endif // DMCL_LEARN_FM_H_

/*!
 *  Copyright (c) 2015 by Contributors
 * \file linear.h
 * \brief Linear and Logistic regression
 *
 * \author Tianqi Chen
 */
#ifndef DMLC_LEARN_LINEAR_H_
#define DMLC_LEARN_LINEAR_H_
#include <omp.h>
#include <dmlc/data.h>
#include "../solver/lbfgs.h"

namespace dmlc {
	namespace linear {
		/*! \brief simple linear model */
		struct LinearModel {
			struct ModelParam {
				/*! \brief global bias */
				float base_score;
				/*! \brief number of features  */
				size_t num_feature;
				/*! \brief loss type*/
				int loss_type;
				// reserved field
				int reserved[16];
				// constructor
				ModelParam(void) {
					memset(this, 0, sizeof(ModelParam));
					base_score = 0.5f;
					num_feature = 0;
					loss_type = 1;
					num_feature = 0;
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
				inline double MarginToPred(double margin) const {
					if (loss_type == 1) {
						return 1.0f / (1.0f + std::exp(-margin));
					} else {
						return margin;
					}
				}
				// margin to loss
				inline double MarginToLoss(double label, double weight, double margin) const {
					if(loss_type ==1) {
						double nlogprob;
						if (margin > 0.0f) {
							nlogprob = std::log(1.0f + std::exp(-margin));
						} else {
							nlogprob = -margin + std::log(1.0f + std::exp(margin));
						}
						return label * weight * nlogprob +
							(1.0f -label) * weight * (margin + nlogprob); 
					}else {
						double diff = margin - label;
						return 0.5f * weight * diff * diff;
					}
				}

				inline double PredToGrad(double label, double weight, double pred) const {
					return weight * (pred - label);
				}
				inline double PredictMargin(const double *weight,
						const Row<unsigned> &v) const {
					// weight[num_feature] is bias
					double sum = base_score + weight[num_feature];
					for (unsigned i = 0; i < v.length; ++i) {
						if (v.index[i] >= num_feature) continue;
						sum += weight[v.index[i]] * v.get_value(i);
					}    
					return sum;
				}

				inline double Predict(const double *weight,
						const Row<unsigned> &v) const {
					return MarginToPred(PredictMargin(weight, v));
				}

			};
			// model parameter
			ModelParam param;
			// weight corresponding to the model
			double *weight;
			LinearModel(void) : weight(NULL) {
			}
			~LinearModel(void) {
				if (weight != NULL) delete [] weight;
			}
			// load model
			inline void Load(dmlc::Stream *fi) {
				fi->Read(&param, sizeof(param));
				if (weight == NULL) {
					weight = new double[param.num_feature + 1];
				}
				fi->Read(weight, sizeof(double) * (param.num_feature + 1));
			}

			inline void Load(dmlc::Stream *fi, const int nfactor) {                                             
				fi->Read(&param, sizeof(param));                                                  
				if (weight == NULL) {                                                            
					weight = new double[param.num_feature + param.num_feature * nfactor + 1];                                    
				}                                                                                
				fi->Read(weight, sizeof(double) * (param.num_feature + param.num_feature * nfactor + 1));                   
			}   

			inline void Save(dmlc::Stream *fo, const int nfactor, const double *wptr = NULL) { 
				fo->Write(&param, sizeof(param));                                                
				if (wptr == NULL) wptr = weight; 
				fo->Write(wptr, sizeof(double) * (param.num_feature + param.num_feature * nfactor + 1)); 
			} 


			inline void Save(dmlc::Stream *fo, const double *wptr = NULL) {
				fo->Write(&param, sizeof(param));
				if (wptr == NULL) wptr = weight;
				fo->Write(wptr, sizeof(double) * (param.num_feature + 1));
			}
			inline double Predict(const Row<unsigned> &v) const {
				return param.Predict(weight, v);
			}
		};
	}  // namespace linear
}  // namespace dmlc
#endif // DMCL_LEARN_LINEAR_H_

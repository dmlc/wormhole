/*!
 * Copyright (c) 2015 by Contributors
 * \file fm.cc
 * \brief Factorization Machines
 *
 */

#include <dmlc/io.h>
#include <dmlc/data.h>
#include <dmlc/logging.h>
//#include "/usr/include/boost/random.hpp"
#include "./fm.h"

namespace dmlc {
namespace fm {
class FmObjFunction : public solver::IObjFunction<float> {
 public:
  // training threads
  int nthread;
  // factor num
  int nfactor;
  // L2 regularization
  float reg_L2;
  // L2 regularization for V
  float reg_L2_fm;
  // fm_random
  float fm_random;
  // model
  FmModel model;
  // training data
  dmlc::RowBlockIter<unsigned> *dtrain;
  // solver
  solver::LBFGSSolver<float> lbfgs;
  // constructor
  FmObjFunction(dmlc::RowBlockIter<unsigned> *dtrain)
      : dtrain(dtrain) {
    lbfgs.SetObjFunction(this);
    nthread = 1;
    nfactor = 8;
    reg_L2 = 1.0f;
    reg_L2_fm = reg_L2;
    fm_random = 0.01f;
    model.weight = NULL;
    task = "train";
    model_in = "NULL";
    name_pred = "pred.txt";
    model_out = "final.model";
  }
  virtual ~FmObjFunction(void) {
    delete dtrain;
  }  
  // set parameters
  inline void SetParam(const char *name, const char *val) {
    model.param.SetParam(name, val);
    lbfgs.SetParam(name, val);
    if (!strcmp(name, "num_feature")) {
      char ndigit[30];
      sprintf(ndigit, "%lu", model.param.num_feature * (nfactor + 1) + 1);
      lbfgs.SetParam("num_dim", ndigit);
    }
    if (!strcmp(name, "reg_L2")) {
      reg_L2 = static_cast<float>(atof(val));
    }
    if (!strcmp(name, "reg_L2_fm")) {
      reg_L2_fm = static_cast<float>(atof(val));
    }
    if (!strcmp(name, "fm_random")) {
      fm_random = static_cast<float>(atof(val));
    }
    if (!strcmp(name, "nthread")) {
      nthread = atoi(val);
    }
    if(!strcmp(name, "nfactor")){
      nfactor = atoi(val);
    }
    if (!strcmp(name, "task")) task = val;
    if (!strcmp(name, "model_in")) model_in = val;
    if (!strcmp(name, "model_out")) model_out = val;
    if (!strcmp(name, "name_pred")) name_pred = val;
  }
  inline void Run(void) {
    if (model_in != "NULL") {
      this->LoadModel(model_in.c_str());
    }
    if (task == "train") {
      lbfgs.Run();
      if (rabit::GetRank() == 0) {
        this->SaveModel(model_out.c_str(), lbfgs.GetWeight());
      }
    } else if (task == "pred") {
      this->TaskPred();
    } else {
      LOG(FATAL) << "unknown task" << task;
    }
  }
  inline void TaskPred(void) {
    CHECK(model_in != "NULL") << "must set model_in for task=pred";
    dmlc::Stream *fo = dmlc::Stream::Create(name_pred.c_str(), "w");
    dmlc::ostream os(fo);
    dtrain->BeforeFirst();
    while (dtrain->Next()) {
      const RowBlock<unsigned> &batch = dtrain->Value();
      for (size_t i = 0; i < batch.size; ++i) {
        os << model.Predict(batch[i], nfactor) << '\n';
      }
    }
    // make sure os push to the output stream, before delete
    os.set_stream(NULL);
    delete fo;
    printf("Finishing writing to %s\n", name_pred.c_str());
  }
  inline void LoadModel(const char *fname) {
    Stream *fi = Stream::Create(fname, "r");
    std::string header; header.resize(4);
    // check header for different binary encode
    CHECK(fi->Read(&header[0], 4) != 0) << "invalid model";
    // base64 format
    if (header == "binf") {
      model.Load(fi, nfactor);
    } else {
      LOG(FATAL) << "invalid model file";
    }
    delete fi;
  }
  inline void SaveModel(const char *fname,
                        const float *wptr,
                        bool save_base64 = false) {
    Stream *fo = Stream::Create(fname, "w");
    fo->Write("binf", 4);
    model.Save(fo, nfactor, wptr);
    delete fo;
  }
  virtual size_t InitNumDim(void)  {
    if (model_in == "NULL") {
      size_t ndim = dtrain->NumCol();
      rabit::Allreduce<rabit::op::Max>(&ndim, 1);
      model.param.num_feature = std::max(ndim, model.param.num_feature);
    }
    return model.param.num_feature * (nfactor + 1) + 1;
  }
  virtual void InitModel(float *weight, size_t size) {
    if (model_in == "NULL") {
    /*  if(rabit::GetRank() == 0){
              boost::mt19937 gen;
              boost::uniform_01<boost::mt19937&> u01(gen);
              boost::normal_distribution<> nd(0,1);
        for(size_t i = 0; i < size; ++i) {
          weight[i] = nd(u01) * fm_random;
        }
      } */
      memset(weight, 0.0f, size * sizeof(float));
      model.param.InitBaseScore();
    } else {
      rabit::Broadcast(model.weight, size * sizeof(float), 0);
      memcpy(weight, model.weight, size * sizeof(float));
    }
  }
  // load model
  virtual void Load(rabit::Stream *fi) {
    fi->Read(&model.param, sizeof(model.param));
  }
  virtual void Save(rabit::Stream *fo) const {
    fo->Write(&model.param, sizeof(model.param));
  }
  virtual double Eval(const float *weight, size_t size) {
    if (nthread != 0) omp_set_num_threads(nthread);
    CHECK(size == model.param.num_feature * (nfactor + 1) + 1);
    double sum_val = 0.0;
    dtrain->BeforeFirst();
    while (dtrain->Next()) {
      const RowBlock<unsigned> &batch = dtrain->Value();
      #pragma omp parallel for schedule(static) reduction(+:sum_val)
      for (size_t i = 0; i < batch.size; ++i) {
        float py = model.param.PredictMargin(weight, batch[i], nfactor);
        float fv = model.param.MarginToLoss(batch[i].label, py);
        sum_val += fv;
      }
    }
    if (rabit::GetRank() == 0) {
      // only add L2 regularization once
      if (reg_L2 != 0.0f) {
        double sum_sqr = 0.0;
        for (size_t i = 0; i < model.param.num_feature; ++i) {
          sum_sqr += weight[i] * weight[i];
        }
        sum_val += 0.5 * reg_L2 * sum_sqr;        
      }
      if (reg_L2_fm != 0.0f) {
        double sum_sqr = 0.0;
        for (size_t i = model.param.num_feature; i < model.param.num_feature * (nfactor + 1); ++i) {
          sum_sqr += weight[i] * weight[i];
        }
        sum_val += 0.5 * reg_L2_fm * sum_sqr;
      }
    }
    CHECK(!std::isnan(sum_val)) << "nan occurs";
    return sum_val;
  }
  virtual void CalcGrad(float *out_grad,
                        const float *weight,
                        size_t size) {
    if (nthread != 0) omp_set_num_threads(nthread);
    
    CHECK(size == model.param.num_feature * (nfactor + 1) + 1) << "size consistency check";
    memset(out_grad, 0.0f, sizeof(float) * size);
    double sum_gbias = 0.0;
    dtrain->BeforeFirst();    
    while (dtrain->Next()) {
      const RowBlock<unsigned> &batch = dtrain->Value();
      static std:: vector<float> tmp(size,0);
      static std:: vector<std::vector<float> > tmp_out_grad(nthread,tmp);
      #pragma omp parallel for schedule(static)
      for (size_t i = 0; i < batch.size; ++i) {
        Row<unsigned> v = batch[i];
        int thread_id = omp_get_thread_num();
        float py = model.param.Predict(weight, v, nfactor);
        float grad = model.param.PredToGrad(v.label, py);
        //add bias
        sum_gbias += grad;
        // single feature grad 
        for (index_t j = 0; j < v.length; ++j) {
          tmp_out_grad[thread_id][v.index[j]] += v.get_value(j) * grad;
        }
        // interation features grad
        for(int i = 0; i < nfactor; ++i) {
          double sumxf = 0.0;
          for(index_t j = 0; j < v.length; ++j){
            int n = model.param.num_feature   + v.index[j] * nfactor + i;
            sumxf += weight[n] * v.get_value(j);
          }
          for(index_t j=0; j < v.length; ++j) {
            int n = model.param.num_feature   + v.index[j] * nfactor + i;
            tmp_out_grad[thread_id][n] += v.get_value(j)  * (sumxf - weight[n] * v.get_value(j)) * grad;
          }
        }
      }
      for(int i = 0; i < nthread;i++){
        #pragma omp parallel for 
        for(size_t j = 0; j < size;j++){
            out_grad[j] += tmp_out_grad[i][j];
            tmp_out_grad[i][j] = 0.0;
        }
      }
     
    }
    out_grad[model.param.num_feature * (nfactor + 1)] = static_cast<float>(sum_gbias);
    if (rabit::GetRank() == 0) {
      // only add L2 regularization once
      if (reg_L2 != 0.0f) {
        for (size_t i = 0; i < model.param.num_feature; ++i) {
          out_grad[i] += reg_L2 * weight[i];
        }
      }
      if (reg_L2_fm != 0.0f) {
        for (size_t i = model.param.num_feature; i < model.param.num_feature * (nfactor + 1); ++i) {
          out_grad[i] += reg_L2_fm * weight[i];
        }
      }
    }
  }
    
 private:
  std::string task;
  std::string model_in;
  std::string model_out;
  std::string name_pred;
};
}  // namespace fm
}  // namespace rabit

int main(int argc, char *argv[]) {
  if (argc < 2) {
    // intialize rabit engine
    rabit::Init(argc, argv);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Usage: <data_in> param=val\n");
    }
    rabit::Finalize();
    return 0;
  }
  rabit::Init(argc, argv);
  dmlc::RowBlockIter<unsigned> *data
      = dmlc::RowBlockIter<unsigned>::Create
      (argv[1],
       rabit::GetRank(),
       rabit::GetWorldSize(),
       "libsvm");  
  dmlc::fm::FmObjFunction *fm = new dmlc::fm::FmObjFunction(data);
  for (int i = 2; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
      fm->SetParam(name, val);
    }
  }
  fm->Run();
  delete fm;
  rabit::Finalize();
  return 0;
}

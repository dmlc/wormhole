#include <dmlc/io.h>
#include <dmlc/data.h>
#include <dmlc/logging.h>
#include "./linear.h"

namespace dmlc {
namespace linear {
class LinearObjFunction : public solver::IObjFunction<float> {
 public:
  // training threads
  int nthread;
  // L2 regularization
  float reg_L2;
  // model
  LinearModel model;
  // training data
  dmlc::RowBlockIter<unsigned> *dtrain;
  // solver
  solver::LBFGSSolver<float> lbfgs;
  // constructor
  LinearObjFunction(dmlc::RowBlockIter<unsigned> *dtrain)
      : dtrain(dtrain) {
    lbfgs.SetObjFunction(this);
    nthread = 1;
    reg_L2 = 0.0f;
    model.weight = NULL;
    task = "train";
    model_in = "NULL";
    name_pred = "pred.txt";
    model_out = "final.model";
  }
  virtual ~LinearObjFunction(void) {
    delete dtrain;
  }  
  // set parameters
  inline void SetParam(const char *name, const char *val) {
    model.param.SetParam(name, val);
    lbfgs.SetParam(name, val);
    if (!strcmp(name, "num_feature")) {
      char ndigit[30];
      sprintf(ndigit, "%lu", model.param.num_feature + 1);
      lbfgs.SetParam("num_dim", ndigit);
    }
    if (!strcmp(name, "reg_L2")) {
      reg_L2 = static_cast<float>(atof(val));
    }
    if (!strcmp(name, "nthread")) {
      nthread = atoi(val);
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
        os << model.Predict(batch[i]) << '\n';
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
      model.Load(fi);
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
    model.Save(fo, wptr);
    delete fo;
  }
  virtual size_t InitNumDim(void)  {
    if (model_in == "NULL") {
      size_t ndim = dtrain->NumCol();
      rabit::Allreduce<rabit::op::Max>(&ndim, 1);
      model.param.num_feature = std::max(ndim, model.param.num_feature);
    }
    return model.param.num_feature + 1;
  }
  virtual void InitModel(float *weight, size_t size) {
    if (model_in == "NULL") {
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
    CHECK(size == model.param.num_feature + 1);
    double sum_val = 0.0;
    dtrain->BeforeFirst();
    while (dtrain->Next()) {
      const RowBlock<unsigned> &batch = dtrain->Value();
      #pragma omp parallel for schedule(static) reduction(+:sum_val)
      for (size_t i = 0; i < batch.size; ++i) {
        float py = model.param.PredictMargin(weight, batch[i]);
        float fv = model.param.MarginToLoss(batch[i].label, py);
        sum_val += fv;
      }
    }
    if (rabit::GetRank() == 0) {
      // only add regularization once
      if (reg_L2 != 0.0f) {
        double sum_sqr = 0.0;
        for (size_t i = 0; i < model.param.num_feature; ++i) {
          sum_sqr += weight[i] * weight[i];
        }
        sum_val += 0.5 * reg_L2 * sum_sqr;        
      }
    }
    CHECK(!std::isnan(sum_val)) << "nan occurs";
    return sum_val;
  }
  virtual void CalcGrad(float *out_grad,
                        const float *weight,
                        size_t size) {
    if (nthread != 0) omp_set_num_threads(nthread);
    
    CHECK(size == model.param.num_feature + 1) << "size consistency check";
    memset(out_grad, 0.0f, sizeof(float) * size);
    double sum_gbias = 0.0;    
    dtrain->BeforeFirst();
    while (dtrain->Next()) {
      const RowBlock<unsigned> &batch = dtrain->Value();
      #pragma omp parallel reduction(+:sum_gbias)
      {
        size_t nfeat = model.param.num_feature;
        size_t npart = omp_get_num_threads();
        size_t step = (nfeat + npart - 1) / npart;
        size_t tid = omp_get_thread_num();
        size_t fbegin = std::min(nfeat, tid * step);
        size_t fend = std::min(nfeat, (tid + 1) * step);
        for (size_t i = 0; i < batch.size; ++i) {
          Row<unsigned> v = batch[i];
          float py = model.param.Predict(weight, v);
          float grad = model.param.PredToGrad(v.label, py);
          for (index_t j = 0; j < v.length; ++j) {
            if (v.index[j] >= fbegin && v.index[j] < fend) {
              out_grad[v.index[j]] += v.get_value(j) * grad;
            }
          }
          sum_gbias += grad;
        }
      }
    }
    out_grad[model.param.num_feature] = static_cast<float>(sum_gbias);
    if (rabit::GetRank() == 0) {
      // only add regularization once
      if (reg_L2 != 0.0f) {
        for (size_t i = 0; i < model.param.num_feature; ++i) {
          out_grad[i] += reg_L2 * weight[i];
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
}  // namespace linear
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
      (dmlc::InputSplit::Create(argv[1],
                                rabit::GetRank(),
                                rabit::GetWorldSize()));  
  dmlc::linear::LinearObjFunction *linear = new dmlc::linear::LinearObjFunction(data);
  for (int i = 2; i < argc; ++i) {
    char name[256], val[256];
    if (sscanf(argv[i], "%[^=]=%s", name, val) == 2) {
      linear->SetParam(name, val);
    }
  }
  linear->Run();
  delete linear;
  rabit::Finalize();
  return 0;
}

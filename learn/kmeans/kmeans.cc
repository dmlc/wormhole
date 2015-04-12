/*!
 * \file kmeans.cc
 * \brief kmeans using rabit allreduce
 */
#include <algorithm>
#include <vector>
#include <cmath>
#include <rabit.h>
#include <dmlc/io.h>
#include <dmlc/data.h>
#include <dmlc/logging.h>

using namespace rabit;
using namespace dmlc;
/*!\brief computes a random number modulo the value */
inline int Random(int value) {
  return static_cast<int>(static_cast<double>(rand()) / RAND_MAX * value);
}

// simple dense matrix, mshadow or Eigen matrix was better
// use this to make a standalone example
struct Matrix {
  inline void Init(size_t nrow, size_t ncol, float v = 0.0f) {
    this->nrow = nrow;
    this->ncol = ncol;
    data.resize(nrow * ncol);
    std::fill(data.begin(), data.end(), v);
  }
  inline float *operator[](size_t i) {
    return &data[0] + i * ncol;
  }
  inline const float *operator[](size_t i) const {
    return &data[0] + i * ncol;
  }
  inline void Print(dmlc::Stream *fo) {
    dmlc::ostream os(fo);
    for (size_t i = 0; i < data.size(); ++i) {
      os << data[i];
      if ((i+1) % ncol == 0) {
        os << '\n';
      } else {
        os << ' ';
      }
    }
  }
  // number of data
  size_t nrow, ncol;
  std::vector<float> data;
};

// kmeans model
class Model : public dmlc::Serializable {
 public:
  // matrix of centroids
  Matrix centroids;
  // load from stream
  virtual void Load(dmlc::Stream *fi) {
    fi->Read(&centroids.nrow, sizeof(centroids.nrow));
    fi->Read(&centroids.ncol, sizeof(centroids.ncol));
    fi->Read(&centroids.data);
  }
  /*! \brief save the model to the stream */
  virtual void Save(dmlc::Stream *fo) const {
    fo->Write(&centroids.nrow, sizeof(centroids.nrow));
    fo->Write(&centroids.ncol, sizeof(centroids.ncol));
    fo->Write(centroids.data);
  }
  virtual void InitModel(unsigned num_cluster, unsigned feat_dim) {
    centroids.Init(num_cluster, feat_dim);
  }
  // normalize L2 norm
  inline void Normalize(void) {
    for (size_t i = 0; i < centroids.nrow; ++i) {
      float *row = centroids[i];
      double wsum = 0.0;
      for (size_t j = 0; j < centroids.ncol; ++j) {
        wsum += row[j] * row[j];
      }
      wsum = sqrt(wsum);
      if (wsum < 1e-6) return;
      float winv = 1.0 / wsum;
      for (size_t j = 0; j < centroids.ncol; ++j) {
        row[j] *= winv;
      }
    }
  }
};
// initialize the cluster centroids
inline void InitCentroids(dmlc::RowBlockIter<unsigned> *data,
                          Matrix *centroids) {
  data->BeforeFirst();
  CHECK(data->Next()) << "dataset is empty";
  const RowBlock<unsigned> &block = data->Value();
  int num_cluster = centroids->nrow;
  for (int i = 0; i < num_cluster; ++i) {
    int index = Random(block.size);
    Row<unsigned> v = block[index];
    for (unsigned j = 0; j < v.length; ++j) {
      (*centroids)[i][v.index[j]] = v.get_value(j);
    }
  }
  for (int i = 0; i < num_cluster; ++i) {
    int proc = Random(rabit::GetWorldSize());
    rabit::Broadcast((*centroids)[i], centroids->ncol * sizeof(float), proc);
  }
}
// calculate cosine distance
inline double Cos(const float *row,
                  const Row<unsigned> &v) {
  double rdot = 0.0, rnorm = 0.0; 
  for (unsigned i = 0; i < v.length; ++i) {
    const dmlc::real_t fv = v.get_value(i);
    rdot += row[v.index[i]] * fv;
    rnorm += fv * fv;
  }
  return rdot  / sqrt(rnorm);
}
// get cluster of a certain vector
inline size_t GetCluster(const Matrix &centroids,
                         const Row<unsigned> &v) {
  size_t imin = 0;
  double dmin = Cos(centroids[0], v);
  for (size_t k = 1; k < centroids.nrow; ++k) {
    double dist = Cos(centroids[k], v);
    if (dist > dmin) {
      dmin = dist; imin = k;
    }
  }
  return imin;
}
             
int main(int argc, char *argv[]) {
  if (argc < 5) {
    // intialize rabit engine
    rabit::Init(argc, argv);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Usage: <data_path> num_cluster max_iter <out_model>\n");
    }
    rabit::Finalize();
    return 0;
  }
  srand(0);    
  // set the parameters
  int num_cluster = atoi(argv[2]);
  int max_iter = atoi(argv[3]);
  // intialize rabit engine
  rabit::Init(argc, argv);
  
  RowBlockIter<index_t> *data
      = RowBlockIter<index_t>::Create
      (InputSplit::Create(argv[1],
                          rabit::GetRank(),
                          rabit::GetWorldSize()));    
  // load model
  Model model; 
  int iter = rabit::LoadCheckPoint(&model);
  if (iter == 0) {
    size_t fdim = data->NumCol();
    rabit::Allreduce<op::Max>(&fdim, 1);
    model.InitModel(num_cluster, fdim);
    InitCentroids(data, &model.centroids);
    model.Normalize();
  }
  const unsigned num_feat = static_cast<unsigned>(model.centroids.ncol);

  // matrix to store the result
  Matrix temp;
  for (int r = iter; r < max_iter; ++r) {
    temp.Init(num_cluster, num_feat + 1, 0.0f);
    auto lazy_get_centroid = [&]()
    {
      // lambda function used to calculate the data if necessary
      // this function may not be called when the result can be directly recovered
      data->BeforeFirst();
      while (data->Next()) {
        const auto &batch = data->Value();
        for (size_t i = 0; i < batch.size; ++i) {
          auto v = batch[i];
          size_t k = GetCluster(model.centroids, v);
          // temp[k] += v
          for (size_t j = 0; j < v.length; ++j) {
            temp[k][v.index[j]] += v.get_value(j);
          }
          // use last column to record counts
          temp[k][num_feat] += 1.0f;
        }
      }
    };
    rabit::Allreduce<op::Sum>(&temp.data[0], temp.data.size(), lazy_get_centroid);
    // set number
    for (int k = 0; k < num_cluster; ++k) {
      float cnt = temp[k][num_feat];
      utils::Check(cnt != 0.0f, "get zero sized cluster");
      for (unsigned i = 0; i < num_feat; ++i) {
        model.centroids[k][i] = temp[k][i] / cnt;
      }
    }
    model.Normalize();
    rabit::LazyCheckPoint(&model);
    if (rabit::GetRank() == 0) {
      rabit::TrackerPrintf("Finish %d-th iteration\n", r);
    }
  }
  delete data;

  // output the model file to somewhere
  if (rabit::GetRank() == 0) {
    auto *fo = Stream::Create(argv[4], "w");
    model.centroids.Print(fo);
    delete fo;
    rabit::TrackerPrintf("All iteration finished, centroids saved to %s\n", argv[4]);
  }
  rabit::Finalize();
  return 0;
}

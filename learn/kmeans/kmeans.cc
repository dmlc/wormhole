/*!
 * \file kmeans.cc
 * \brief kmeans using rabit allreduce
 */
#define OMP_DBG
#include <algorithm>
#include <vector>
#include <cmath>
#include <rabit.h>
#include <dmlc/io.h>
#include <dmlc/data.h>
#include <dmlc/logging.h>
#include <omp.h>

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
                         const Row<unsigned> &v, double* out_dist = NULL) {
  size_t imin = 0;
  double dmin = Cos(centroids[0], v);
  for (size_t k = 1; k < centroids.nrow; ++k) {
    double dist = Cos(centroids[k], v);
    if (dist > dmin) {
      dmin = dist; imin = k;
    }
  }
  if (out_dist)
    *out_dist = dmin;
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
  int omp_num = atoi(argv[5]);
  // intialize rabit engine
  rabit::Init(argc, argv);
  
  RowBlockIter<index_t> *data
      = RowBlockIter<index_t>::Create
      (argv[1],
       rabit::GetRank(),
       rabit::GetWorldSize(),
       "libsvm");
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
      double dist_sum = 0;
      double dist_tmp;
      data->BeforeFirst();
      while (data->Next()) {
        const auto &batch = data->Value();
        for (size_t i = 0; i < batch.size; ++i) {
          auto v = batch[i];
          size_t k = GetCluster(model.centroids, v, &dist_tmp);
          dist_sum += dist_tmp;
          // temp[k] += v
          for (size_t j = 0; j < v.length; ++j) {
            temp[k][v.index[j]] += v.get_value(j);
          }
          // use last column to record counts
          temp[k][num_feat] += 1.0f;
        }
      }
      printf("total dist = %lf\n", dist_sum);
    };
    auto omp_get_centroid = [&]()
    {
      // lambda function used to calculate the data if necessary
      // this function may not be called when the result can be directly recovered
      std::vector<int> cid;
      data->BeforeFirst();
      while (data->Next()) {
        const auto &batch = data->Value();
        size_t batch_size = batch.size;
        cid.resize(batch_size);

        // get cluster_id for each instance, write to vector cid
        #pragma omp parallel num_threads(omp_num) 
        {
          #pragma omp for
          for (size_t i = 0; i < batch_size; ++i) {
            if (i >= batch_size)
              continue;
            int k = GetCluster(model.centroids, batch[i]);
            cid[i] = k;
          }
        }
        //group instance by cluster_id
        std::vector<std::vector<int> > instances_cid;
        instances_cid.resize(num_cluster);
        for (size_t i=0; i<batch_size; i++) 
          instances_cid[cid[i]].push_back(i);
        /*
        for (int i=0; i<num_cluster; i++)
          std::cout << instances_cid[i].size() << " ";
        std::cout <<"\n";
        */

        // compute centroid for each cluster
        #pragma omp parallel for num_threads(omp_num) schedule(dynamic)
        for (int k = 0; k < num_cluster; k++) {
          if (k >= num_cluster)
            continue;
          for (size_t idx = 0; idx < instances_cid[k].size(); idx++) {
            int i = instances_cid[k][idx];
            auto v = batch[i];
            // temp[k] += v
            for (size_t j = 0; j < v.length; ++j) {
              temp[k][v.index[j]] += v.get_value(j);
            }
            // use last column to record counts
            temp[k][num_feat] += 1.0f;
          }
        }
      }
      //printf("total dist = %lf\n", dist_sum);
    };
    auto get_centroid = omp_get_centroid;
    rabit::Allreduce<op::Sum>(&temp.data[0], temp.data.size(), get_centroid);
    //printf("num_cluster = %d\n", num_cluster);
    // set number
    for (int k = 0; k < num_cluster; ++k) {
      float cnt = temp[k][num_feat];
      if (cnt != 0.0f) {        
        for (unsigned i = 0; i < num_feat; ++i) {
          model.centroids[k][i] = temp[k][i] / cnt;
        }
      } else {
        rabit::TrackerPrintf("Error: found zero size cluster, maybe too less number of datapoints?\n");
        exit(-1);
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

/**
 * @file   async_sgd.h
 * @brief  Asynchronous stochastic gradient descent to solve linear methods.
 */
#include "./linear.h"
#include "proto/linear.pb.h"
#include "base/minibatch_iter.h"

namespace dmlc {
namespace linear {

// #include "../../repo/ps-lite/src/base/blob.h"
// using ps::Key FeaID;

// commands
static const int kRequestWorkload = 1;


/***************************************
 * \brief A server node
 **************************************/
template <typename V>
class AsyncSGDServer : public ISGDCompNode {
 public:

};

/***************************************
 * \brief A worker node
 **************************************/
template <typename V>
class AsyncSGDWorker : public ISGDCompNode {
 public:
  AsyncSGDWorker(const Config& conf) : conf_(conf) {
      // : ISGDCompNode(), conf_(conf) {
    // loss_ = createLoss<V>(conf_.loss());
  }
  virtual ~AsyncSGDWorker() { }

  virtual void Run() {
    while (true) {
      // request one training data file from the scheduler
      Task task; task.set_cmd(kRequestWorkload);
      Wait(Submit(task, SchedulerID()));
      std::string file = LastResponse()->msg();

      // train
      if (file.empty()) {
        LOG(INFO) << MyNodeID() << ": all workloads are done";
        break;
      }
      Train(file);
    }
  }

 private:
  using Minibatch = data::RowBlockContainer<unsigned>;
  using SharedFeaID = std::shared_ptr<std::vector<FeaID> >;

  void Train(const std::string file) {
    LOG(INFO) << MyNodeID() << ": start to process " << file;
    dmlc::data::MinibatchIter<FeaID> iter(
        file, 0, 1, conf_.data_format(), conf_.minibatch());
    iter.BeforeFirst();

    int id = 0;
    while (iter->Next()) {
      // find the feature id in this minibatch
      Minibatch* batch = new Minibatch();
      SharedFeaID feaid(new std::vector<FeaID>());
      Localizer<FeaID> lc;
      lc.Localize(iter->Value(), feaid.get(), batch);

      // pull the weight for this minibatch from servers
     std::vector<real_t>* w = new std::vector<real_t>(feaid.size());
      SyncOpts opts;
      opts.callback = [this, batch, feaid, w]() {
        CalcGrad(*batch, feaid, *w);
        delete batch;
        delete w;
      };
      ps_->ZPull(feaid, w, opts);
    }
    LOG(INFO) << MyNodeID() << ": finished " << file;
  }

  /**
   * @brief Compute gradient
   * @param id minibatch id
   */
  void CalcGrad(const Minibatch& batch, const SharedFeaID& feaid,
                const std::vector<real_t> weight) {
    // evalute

    // compute gradient

    // push

  }

  Config conf_;
  KVWorker<FeaID> ps_;


};


}  // namespace linear
}  // namespace dmlc

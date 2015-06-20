#pragma once

// #include "base/dist_monitor.h"
#include "base/minibatch_iter.h"
namespace dmlc {
namespace fm {

using FeaID = ps::Key;
using Real = float;
template <typename T> using Blob = ps::Blob<T>;

// commands
static const int kProcess = 1;
static const int kSaveModel = 2;
static const int kPushFeaCnt = 3;

////////////////////////////////////////////////////////////
//   model
////////////////////////////////////////////////////////////

// #pragma pack(push)
// #pragma pack(4)
// #pragma pack(pop)

struct AdaGradEntry {
  AdaGradEntry() { }
  ~AdaGradEntry() { if ( size > 1 ) { delete [] w; delete [] sq_cum_grad; } }

  // appearence of this feature in the seened data
  unsigned fea_cnt = 0;

  // length of w. if size == 1, then use w itself to store the value
  unsighed size = 1;

  Real *w = NULL;
  Real *sq_cum_grad = NULL;
};

////////////////////////////////////////////////////////////
//  model updater
////////////////////////////////////////////////////////////


/**
 * \brief the base handle class
 */
struct ISGDHandle {
  inline void Start(bool push, int timestamp, int cmd, void* msg) {
    push_count = (push && (cmd == kPushFeaCnt)) ? true : false;
  }
  inline void Finish() { tracker->Report(); }
  inline void SetCaller(void *obj) { }

  bool push_count;
  ModelMonitor* tracker = nullptr;

  Real alpha = 0.1, beta = 1;
  Real theta = 1;

  int fea_thr = 100;
  int fea_thr2 = 1000000;

  int k = 50;
  int k2 = 1000;
};

struct AdaGradHandle : public ISGDHandle {
  inline void Push(FeaID key, Blob<const Real> recv, AdaGradEntry& val) {
    if (push_count) {
      val.fea_cnt += (unsigned) recv[0];
      if (val.fea_cnt > fea_thr2 && val.size < k2) {
        Resize(k2 + 1, val);
      } else if (val.fea_cnt > fea_thr && val.size < k) {
        Resize(k + 1, val);
      }
    } else {
      CHECK_LE(recv.size, val.size);
      if (val.size == 1) {
        Update((Real)val.w, (Real)val.sq_cum_grad, recv[0]);
      } else {
        for (size_t i = 0; i < recv.size; ++i) {
          Update(val.w[i], val.sq_cum_grad[i], recv[i]);
        }
      }
    }
  }

  inline void Pull(FeaID key, const AdaGradEntry& val, Blob<Real>& send) {
    if (val.size == 1) {
      send[0] = (Real) val.size;
      send.size = 1;
    } else {
      send.data = val.w;
      send.size = val.size;
    }
  }

  inline Update(Real& w, Real& cg, Real g) {
    cg = sqrt(cg*cg + g*g);
    Real eta = (cg + this->beta) / this->alpha;
    w = w - eta * g;
  }

  inline Resize(int n, AdaGradEntry& val) {
    Real* new_w = new Real[n]; memset(new_w, 0, sizeof(Real)*n);
    Real* new_cg = new Real[n]; memset(new_cg, 0, sizeof(Real)*n);

    if (val.size == 1) {
      new_w[0] = (Real) val.w;
      new_cg[0] = (Real) val.sq_cum_grad;
    } else {
      memcpy(new_w, val.w, val.size * sizeof(Real));
      memcpy(new_gc, val.sq_cum_grad, val.size * sizeof(Real));
      delete [] val.w;
      delete [] val.sq_cum_grad;
    }

    val.w = new_w;
    val.sq_cum_grad = new_cg;
  }
};

////////////////////////////////////////////////////////////
//  objective
////////////////////////////////////////////////////////////

class Objective {
 public:

  void Init() {

  }
  /*! \brief evaluate the objective value */
  Real Objv() {
  }

  /*! \brief compute the gradients */
  void CalcGrad() {

  }

 private:

};

////////////////////////////////////////////////////////////
//  sgd solver
////////////////////////////////////////////////////////////

class AsyncSGDWorker : public ps::App {
 public:
  AsyncSGDWorker(const Config& conf) : conf_(conf), reporter_(conf_.disp_itv()) { }
  virtual ~AsyncSGDWorker() { }

  // process request from the scheduler node
  virtual void ProcessRequest(ps::Message* request) {
    int cmd = request->task.cmd();
    if (cmd == kProcess) {
      Workload wl; CHECK(wl.ParseFromString(request->task.msg()));
      if (wl.file_size() < 1) return;
      Process(wl.file(0), wl.type());
      if (wl.type() != Workload::TRAIN) {
        // return the progress
        std::string prog_str; monitor_.prog.Serialize(&prog_str);
        ps::Task res; res.set_msg(prog_str);
        Reply(request, res);
      }
    }
  }

 private:

  void Process(const File& file, Workload::Type type) {
    // use a large minibatch size and max_delay for val or test tasks
    int mb_size = type == Workload::TRAIN ? conf_.minibatch() :
                  std::max(conf_.minibatch()*10, 100000);
    int max_delay = type == Workload::TRAIN ? conf_.max_delay() : 100000;
    num_mb_fly_ = num_mb_done_ = 0;

    LOG(INFO) << ps::MyNodeID() << ": start to process " << file.ShortDebugString();
    dmlc::data::MinibatchIter<FeaID> reader(
        file.file().c_str(), file.k(), file.n(),
        conf_.data_format().c_str(), mb_size);
    reader.BeforeFirst();
    while (reader.Next()) {
      using std::vector;
      using std::shared_ptr;
      using Minibatch = dmlc::data::RowBlockContainer<unsigned>;

      // find all feature id in this minibatch, and convert it to a more compact format
      auto global = reader.Value();
      Minibatch* local = new Minibatch();
      shared_ptr<vector<FeaID> > feaid(new vector<FeaID>());
      shared_ptr<vector<Real> > feacnt(new vector<Real>());

      Localizer<FeaID> lc; lc.Localize(global, local, feaid.get(), feacnt.get());

      // wait for data consistency
      WaitMinibatch(max_delay);

      // push the feature count to the servers
      ps::SyncOpts push_opts;
      SetFilters(true, &push_opts);
      int push_t = server_.ZPush(feaid, feacnt, push_opts);

      // pull the weight from the servers
      vector<Real>* val = new vector<Real>();
      vector<int>* len_val = new vector<Real>(feaid.get()->size());
      ps::SyncOpts opts;

      // this callback will be called when the weight has been actually pulled back
      opts.callback = [this, local, feaid, val, len_val, type]() {
        // eval the progress
        Objective obj;
        obj.Init(local->GetBlock(), *val, *len_val);

        // monitor_.Update(local->label.size(), loss);

        if (type == Workload::TRAIN) {
          // reporting from time to time
          reporter_.Report(0, &monitor_.prog);

          // calculate and push the gradients
          loss->CalcGrad(val, len_val);
          ps::SyncOpts opts;
          // this callback will be called when the gradients have been actually pushed
          opts.callback = [this]() { FinishMinibatch(); };
          // filters to reduce network traffic
          SetFilters(true, &opts);
          server_.ZPush(feaid, shared_ptr<vector<Real>>(val),
                        shared_ptr<vector<int>>(len_val), opts);
        } else {
          // don't need to cal grad for evaluation task
          FinishMinibatch();
          delete buf;
        }
        delete local;
        delete loss;
      };

      // filters to reduce network traffic
      SetFilters(false, &opts);
      opts.deps.push_push(push_t);
      server_.ZPull(feaid, val, val_len, opts);

      mb_mu_.lock(); ++ num_mb_fly_; mb_mu_.unlock();

      LOG(INFO) << "#minibatches on processing: " << num_mb_fly_;
    }

    // wait untill all are done
    WaitMinibatch(0);
    LOG(INFO) << ps::MyNodeID() << ": finished";
  }

  // wait if the currenta number of on processing minibatch > num
  inline void WaitMinibatch(int num) {
    std::unique_lock<std::mutex> lk(mb_mu_);
    mb_cond_.wait(lk, [this, num] {return num_mb_fly_ <= num;});
  }

  inline void FinishMinibatch() {
    // wake the main thread
    mb_mu_.lock();
    -- num_mb_fly_;
    ++ num_mb_done_;
    mb_mu_.unlock();
    mb_cond_.notify_one();
  }

  void SetFilters(bool push, ps::SyncOpts* opts) {
    if (conf_.fixed_bytes() > 0) {
      opts->AddFilter(ps::Filter::FIXING_FLOAT)->set_num_bytes(conf_.fixed_bytes());
    }
    if (conf_.key_cache()) {
      opts->AddFilter(ps::Filter::KEY_CACHING)->set_clear_cache(push);
    }
    if (conf_.msg_compression()) {
      opts->AddFilter(ps::Filter::COMPRESSING);
    }
  }

  Config conf_;
  ps::KVWorker<Real> server_;
  WorkerMonitor monitor_;
  TimeReporter reporter_;

  int num_mb_fly_;
  int num_mb_done_;
  std::mutex mb_mu_;
  std::condition_variable mb_cond_;
};



}  // namespace fm
}  // namespace dmlc

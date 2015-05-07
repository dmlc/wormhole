#include "../base/minibatch_iter.h"
#include "dmlc/timer.h"

int main(int argc, char *argv[]) {
  if (argc < 6) {
    printf("Usage: <libsvm> partid npart format minibatch\n");
    return 0;
  }

  using namespace dmlc;
  data::MinibatchIter<unsigned> reader(
      argv[1], atoi(argv[2]), atoi(argv[3]), argv[4], atoi(argv[5]));

  reader.BeforeFirst();
  size_t num_ex = 0;

  double tv = GetTime();
  while (reader.Next()) {
    auto blk = reader.Value();
    num_ex += blk.size;
    double t = GetTime() - tv;
    LOG(INFO) << "mb " << blk.size
              << ", #index " << blk.offset[blk.size]
              << ", #ex " << num_ex << ", "
              << reader.BytesRead() / 1e6 / t  << " MB/sec, "
              << num_ex / t << " #ex/sec.";
  }
  return 0;
}

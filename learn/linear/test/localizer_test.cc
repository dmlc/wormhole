#include <iostream>
#include "base/localizer.h"
#include "base/utils.h"

int main(int argc, char *argv[]) {
  using namespace dmlc;
  data::RowBlockContainer<unsigned> in, out;
  in.label = {1,2,3};
  in.offset = {0,1,3,5};
  in.index = {100,2,10000,2345,1873};
  in.value = {.1,.2,3,4,5};

  Localizer<unsigned> lc;
  lc.Localize(in.GetBlock(), &out);
  std::cout << "before: \n" << DebugStr(in) << std::endl;
  std::cout << "after: \n" << DebugStr(out) << std::endl;
  return 0;
}

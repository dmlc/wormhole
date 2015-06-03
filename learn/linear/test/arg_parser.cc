#include "base/arg_parser.h"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: conf_file\n");
    return 0;
  }
  using namespace dmlc;
  ArgParser arg;
  arg.ReadFile(argv[1]);
  arg.ReadArgs(argc-2, argv+2);

  return 0;
}

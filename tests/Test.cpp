#include "Test.h"

#include "internal/common/math/math_utils.h"
#include "internal/debug/Logger.h"
#include <gtest/gtest.h>
int main(int argv, char **argc) {
  LoggerConfigDataStruct config;
  std::shared_ptr<std::ostream> out(&std::cout, [](std::ostream *) {});
  config.write_destination = out;
  config.enable_logging = true;
  LOGCONFIG(config);
  LOGDISABLE();
  ::testing::InitGoogleTest(&argv, argc);
  auto a = RUN_ALL_TESTS();
  return a;
}
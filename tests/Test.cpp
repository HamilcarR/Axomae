#include "Test.h"
#include <gtest/gtest.h>

int main(int argv, char **argc) {
  ::testing::InitGoogleTest(&argv, argc);
  auto a = RUN_ALL_TESTS();
  return a;
}
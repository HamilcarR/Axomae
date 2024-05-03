#include "nova_utils.h"
#include "Test.h"

TEST(NovaUtils, divideByTiles) {
  std::vector<nova::Tile> tiles = nova::divideByTiles(50, 50, 5, 5);
  EXPECT_EQ(tiles.size(), 25);
}
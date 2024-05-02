
#ifndef NOVA_UTILS_H
#define NOVA_UTILS_H
#include <vector>

namespace nova {
  struct Tile {
    int width_start;
    int height_start;
    int width_end;
    int height_end;
  };

  /*Divide thread load for a 2D array by tiles*/
  inline std::vector<Tile> divideByTiles(int width, int height, int thread_numbers) {
    if (thread_numbers == 0 || thread_numbers == 1)
      return {{0, 0, width - 1, height - 1}};
    std::vector<Tile> tiles;
    int width_per_thread = width / thread_numbers;
    int remain = width % thread_numbers;
    int i = 0;
    for (i = 0; i < width - remain; i += width_per_thread)  // wrong condition
      tiles.push_back({i, 0, i + width_per_thread, height - 1});
    if (remain != 0)
      tiles.push_back({i, 0, i + remain - 1, height - 1});
    return tiles;
  }
#pragma GCC push_options
#pragma GCC optimize("O0")
  inline std::vector<Tile> divideByTiles(int total_width, int total_height, int N_tiles_W, int N_tiles_H) {
    std::vector<Tile> tiles;
    int W = total_width / N_tiles_W;
    int H = total_height / N_tiles_H;
    int W_R = W + (total_width % W);
    int H_R = H + (total_height % H);

    for (int j = 0; j < N_tiles_H; ++j) {
      for (int i = 0; i < N_tiles_W; ++i) {
        Tile tile{};
        tile.width_start = i * W;
        tile.width_end = tile.width_start + ((i == (N_tiles_W - 1)) ? W_R : W);
        tile.height_start = j * H;
        tile.height_end = tile.height_start + ((j == (N_tiles_H - 1)) ? H_R : H);
        tiles.push_back(tile);
      }
    }

    return tiles;
  }
#pragma GCC pop_options
}  // namespace nova
#endif  // NOVA_UTILS_H

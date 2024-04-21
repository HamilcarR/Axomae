#ifndef THREAD_UTILS_H
#define THREAD_UTILS_H

#include <vector>

namespace threading {

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
}  // namespace threading
#endif  // THREAD_UTILS_H

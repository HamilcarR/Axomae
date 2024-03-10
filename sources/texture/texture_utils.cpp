#include "texture_utils.h"

namespace utils::texture {
  std::vector<float> create_furnace(int w, int h) { return std::vector<float>(w * h, 1.f); }
}  // namespace utils::texture
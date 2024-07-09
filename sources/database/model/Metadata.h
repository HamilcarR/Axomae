#ifndef METADATA_H
#define METADATA_H
#include "constants.h"

namespace image {
  struct Metadata {
    std::string name{};
    /* File format */
    std::string format{};
    unsigned int height{};
    unsigned int width{};
    unsigned int channels{};
    bool color_corrected{};
    bool is_hdr{};
  };
}  // namespace image
#endif  // AXOMAE_METADATA_H

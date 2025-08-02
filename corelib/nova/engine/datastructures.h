#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H
#include <cstddef>
namespace nova {

  /** Must point to a valid managed buffer */
  template<class T>
  struct RenderBuffers {
    T *accumulator_buffer;
    T *partial_buffer;
    T *depth_buffer;
    size_t color_buffers_pitch{};
    size_t depth_buffers_pitch{};
    size_t byte_size_color_buffers{};
    size_t byte_size_depth_buffers{};
    int channels{};
  };
  using HdrBufferStruct = RenderBuffers<float>;

}  // namespace nova
#endif  // DATASTRUCTURES_H

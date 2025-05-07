#ifndef DATASTRUCTURES_H
#define DATASTRUCTURES_H
#include "camera/nova_camera.h"
#include "material/NovaMaterials.h"
#include "primitive/PrimitiveInterface.h"
#include "shape/ShapeInterface.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/TextureContext.h"
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

#include "device_internal.h"
#include <glm/vec4.hpp>

namespace nova::aggregate {
  extern "C" __constant__ _device_params_s_ parameters;

  __device__ __forceinline__ void shade(render_buffer_t &render_buffer, float u, float v, glm::vec4 color) {
    unsigned px = math::texture::uvToPixel(u, render_buffer.width);
    unsigned py = math::texture::uvToPixel(v, render_buffer.height);
    unsigned offset = (py * render_buffer.width + px) * 4;
    render_buffer.render_target[offset] = color.r;
    render_buffer.render_target[offset + 1] = color.g;
    render_buffer.render_target[offset + 2] = color.b;
    render_buffer.render_target[offset + 3] = color.a;
  }
  __device__ __forceinline__ void shade(render_buffer_t &render_buffer, unsigned x, unsigned y, glm::vec4 color) {
    unsigned offset = (y * render_buffer.width + x) * 4;
    render_buffer.render_target[offset] = color.r;
    render_buffer.render_target[offset + 1] = color.g;
    render_buffer.render_target[offset + 2] = color.b;
    render_buffer.render_target[offset + 3] = color.a;
  }

  __device__ __forceinline__ void shade(render_buffer_t &render_buffer, unsigned offset, glm::vec4 color) {
    render_buffer.render_target[offset] = color.r;
    render_buffer.render_target[offset + 1] = color.g;
    render_buffer.render_target[offset + 2] = color.b;
    render_buffer.render_target[offset + 3] = color.a;
  }

  extern "C" __global__ void __raygen__main() {}

  extern "C" __global__ void __miss__sample_envmap() {}

  extern "C" __global__ void __anyhit__random_intersect() {}

  extern "C" __global__ void __closesthit__minimum_intersect() {}

}  // namespace nova::aggregate

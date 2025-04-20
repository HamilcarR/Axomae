#include "device_internal.h"
#include "engine/datastructures.h"
namespace nova::aggregate {
  extern "C" __constant__ _device_params_s_ parameters;

  __device__ __forceinline__ void shade(_render_buffer_s_ &render_buffer, float u, float v, glm::vec4 color) {
    unsigned px = math::texture::uvToPixel(u, render_buffer.width);
    unsigned py = math::texture::uvToPixel(v, render_buffer.height);
    unsigned offset = (py * render_buffer.width + px) * 4;
    render_buffer.render_target[offset] = color.r;
    render_buffer.render_target[offset + 1] = color.g;
    render_buffer.render_target[offset + 2] = color.b;
    render_buffer.render_target[offset + 3] = color.a;
  }
  __device__ __forceinline__ void shade(_render_buffer_s_ &render_buffer, unsigned x, unsigned y, glm::vec4 color) {
    unsigned offset = (y * render_buffer.width + x) * 4;
    render_buffer.render_target[offset] = color.r;
    render_buffer.render_target[offset + 1] = color.g;
    render_buffer.render_target[offset + 2] = color.b;
    render_buffer.render_target[offset + 3] = color.a;
  }

  __device__ __forceinline__ void shade(_render_buffer_s_ &render_buffer, unsigned offset, glm::vec4 color) {
    render_buffer.render_target[offset] = color.r;
    render_buffer.render_target[offset + 1] = color.g;
    render_buffer.render_target[offset + 2] = color.b;
    render_buffer.render_target[offset + 3] = color.a;
  }

  extern "C" __global__ void __raygen__main() {}

  extern "C" __global__ void __miss__sample_envmap() {}

  extern "C" __global__ void __anyhit__random_intersect() {}

  extern "C" __global__ void __closesthit__minimum_intersect() {}

  void prepare(HdrBufferStruct *render_buffers,
               const gputils::gpu_util_structures_t &gpu_structures,
               const shape::MeshBundleViews &mesh_bundle_views,
               const axstd::span<const material::NovaMaterialInterface> &materials_view,
               const axstd::span<const primitive::NovaPrimitiveInterface> &primitives_view,
               const texturing::TextureBundleViews &texture_bundle_views) {
    _integrator_args_s_ arguments;
    arguments.generator = gpu_structures.random_generator;
    arguments.geometry_context = shape::MeshCtx(mesh_bundle_views);
    arguments.materials = materials_view;
    arguments.primitives = primitives_view;
    arguments.texture_context = texture_bundle_views;
  }

}  // namespace nova::aggregate

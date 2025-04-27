#include "device_internal.h"
#include "gpu/optix_params.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/TextureContext.h"
#include <internal/common/math/math_texturing.h>
#include <internal/device/gpgpu/device_macros.h>
#include <optix_device.h>

#define ax_index optixGetLaunchIndex()
#define ax_dim optixGetLaunchDimensions()

namespace nv_mat = nova::material;
namespace nv_shape = nova::shape;
namespace nv_prim = nova::primitive;
namespace nv_tex = nova::texturing;
using namespace math::texture;

ax_device_force_inlined void shade(render_buffer_s &render_buffer, float u, float v, const glm::vec4 &color) {
  unsigned px = math::texture::uvToPixel(u, render_buffer.width);
  unsigned py = math::texture::uvToPixel(v, render_buffer.height);
  unsigned offset = (py * render_buffer.width + px) * 4;
  render_buffer.render_target[offset] = color.r;
  render_buffer.render_target[offset + 1] = color.g;
  render_buffer.render_target[offset + 2] = color.b;
  render_buffer.render_target[offset + 3] = color.a;
}
ax_device_force_inlined void shade(render_buffer_s &render_buffer, unsigned x, unsigned y, const glm::vec4 &color) {
  unsigned offset = (y * render_buffer.width + x) * 4;
  render_buffer.render_target[offset] = color.r;
  render_buffer.render_target[offset + 1] = color.g;
  render_buffer.render_target[offset + 2] = color.b;
  render_buffer.render_target[offset + 3] = color.a;
}

ax_device_force_inlined void shade(render_buffer_s &render_buffer, unsigned offset, const glm::vec4 &color) {
  render_buffer.render_target[offset] = color.r;
  render_buffer.render_target[offset + 1] = color.g;
  render_buffer.render_target[offset + 2] = color.b;
  render_buffer.render_target[offset + 3] = color.a;
}
extern "C" {
extern ax_device_const nova::optix_traversal_param_s parameters;
}

ax_device_force_inlined nova::device_traversal_param_s get_params() { return parameters.d_params; }

extern "C" ax_kernel void __raygen__main() {
  render_buffer_s rb = {};
  nova::device_traversal_param_s params = get_params();
  rb.render_target = params.render_buffers.partial_buffer;
  rb.width = params.width;
  rb.height = params.height;
  uint3 idx = ax_index;
  uint3 dim = ax_dim;
  float u = (float)idx.x / (float)dim.x;
  float v = (float)idx.y / (float)dim.y;
  nv_tex::TextureCtx ctx = params.texture_bundle_views;
  nv_tex::texture_data_aggregate_s tex_data = {};
  tex_data.texture_ctx = &ctx;
  nv_tex::ImageTexture img(2);
  glm::vec4 pixel = img.sample(u, v, tex_data);
  shade(rb, u, v, pixel);
}
extern "C" ax_kernel void __miss__sample_envmap() {}
extern "C" ax_kernel void __anyhit__random_intersect() {}
extern "C" ax_kernel void __closesthit__minimum_intersect() {}

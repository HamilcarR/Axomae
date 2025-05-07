#include "device_internal.h"
#include "gpu/optix_params.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/TextureContext.h"
#include <internal/common/math/math_camera.h>
#include <internal/common/math/math_texturing.h>
#include <internal/device/gpgpu/device_macros.h>
#include <optix_device.h>

#define ax_index optixGetLaunchIndex()
#define ax_dim optixGetLaunchDimensions()
#define ax_wray_dir optixGetWorldRayDirections()

namespace nv_mat = nova::material;
namespace nv_shape = nova::shape;
namespace nv_prim = nova::primitive;
namespace nv_tex = nova::texturing;
using namespace math::texture;

ax_device_force_inlined void shade(render_buffer_s &render_buffer, float u, float v, const glm::vec4 &color) {
  unsigned px = math::texture::uvToPixel(u, render_buffer.width - 1);
  unsigned py = math::texture::uvToPixel(v, render_buffer.height - 1);
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

ax_device_force_inlined const nova::device_traversal_param_s &get_params() { return parameters.d_params; }

extern "C" ax_kernel void __raygen__main() {
  nova::device_traversal_param_s params = get_params();
  uint3 idx = ax_index;
  uint3 dim = ax_dim;
  float u = (float)idx.x / (float)dim.x;
  float v = (float)idx.y / (float)dim.y;
  const glm::vec2 ndc = math::camera::screen2ndc(idx.x, idx.y, dim.x, dim.y);
  math::camera::camera_ray c_ray = math::camera::ray_inv_mat(ndc.x, ndc.y, params.camera.inv_P, params.camera.inv_V);
  float3 origin = {c_ray.near.x, c_ray.near.y, c_ray.near.z};
  float3 direction = {c_ray.far.x, c_ray.far.y, c_ray.far.z};
  unsigned p0, p1, p2, p3, p4, p5, p6, p7;
  optixTrace(0, origin, direction, 0.001f, 1e30f, OptixVisibilityMask(1), 0, OPTIX_RAY_FLAG_NONE, 0, 0, 0);
}
extern "C" ax_kernel void __miss__sample_envmap() {

  nova::device_traversal_param_s params = get_params();
  uint3 idx = ax_index;
  uint3 dim = ax_dim;
  float u = (float)idx.x / (float)dim.x;
  float v = (float)idx.y / (float)dim.y;
  render_buffer_s rb = {};
  rb.render_target = params.render_buffers.partial_buffer;
  rb.width = params.width;
  rb.height = params.height;
  nova::texturing::texture_data_aggregate_s data = {};
  float3 dir = {0.f, 1.f, 0.f};
  data.geometric_data.sampling_vector = {dir.x, dir.y, dir.z};
  glm::vec4 pixel = params.environment_map.sample(u, v, data);
  shade(rb, u, v, glm::vec4(1.f));
}
extern "C" ax_kernel void __anyhit__random_intersect() {}
extern "C" ax_kernel void __closesthit__minimum_intersect() {}
extern "C" ax_kernel void __exception__exception_handler() {

  unsigned int type = optixGetExceptionCode();

  printf("Optix device exception : code=%u\n", type);
}

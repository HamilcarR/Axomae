#include "common_routines.cuh"
#include "gpu/optix_params.h"
#include <optix_device.h>

#define ax_index optixGetLaunchIndex()
#define ax_dim optixGetLaunchDimensions()
#define ax_wray_dir optixGetWorldRayDirection()
#define ax_wray_ori optixGetWorldRayOrigin()

template<typename... Args>
ax_device_force_inlined void AX_THROW(unsigned code, Args... detail) {
  optixThrowException(code, detail...);
}

namespace nv_mat = nova::material;
namespace nv_shape = nova::shape;
namespace nv_prim = nova::primitive;
namespace nv_tex = nova::texturing;
using namespace math::texture;

extern "C" {
extern ax_device_const nova::optix_traversal_param_s parameters;
}

ax_device_force_inlined const nova::device_traversal_param_s &get_params() { return parameters.d_params; }

extern "C" ax_kernel void __raygen__main() {
  AX_ASSERT_NOTNULL(parameters.handle);
  nova::device_traversal_param_s params = get_params();
  uint3 idx = ax_index;
  uint3 dim = ax_dim;
  float u = (float)idx.x / (float)dim.x;
  float v = (float)idx.y / (float)dim.y;
  const glm::vec2 ndc = math::camera::screen2ndc(idx.x, params.height - idx.y, params.width, params.height);
  math::camera::camera_ray c_ray = math::camera::ray_inv_mat(ndc.x, ndc.y, params.camera.inv_P, params.camera.inv_V);
  float3 origin = {c_ray.near.x, c_ray.near.y, c_ray.near.z};
  float3 direction = {c_ray.far.x, c_ray.far.y, c_ray.far.z};
  optixTrace(parameters.handle, origin, direction, 0.1f, 1e30f, 0.f, OptixVisibilityMask(0xFF), OPTIX_RAY_FLAG_NONE, 0, 0, 0);
}

template<class T>
ax_device_force_inlined float int2float(T value) {
  return math::texture::rgb_uint2float(value);
}

extern "C" ax_kernel void __miss__sample_envmap() {
  nova::device_traversal_param_s params = get_params();
  uint3 idx = ax_index;
  uint3 dim = ax_dim;
  float u = (float)idx.x / (float)dim.x;
  float v = (float)idx.y / (float)dim.y;

  float3 dir = ax_wray_dir;
  float3 ori = ax_wray_ori;
  nova::Ray ray({ori.x, ori.y, ori.z}, {dir.x, dir.y, dir.z});

  render_buffer_s rb = {};
  rb.render_target = params.render_buffers.partial_buffer;
  rb.width = params.width;
  rb.height = params.height;
  shade(rb, u, v, sample_envmap(ray, &params));
}
extern "C" ax_kernel void __anyhit__random_intersect() {
  uint3 idx = ax_index;
  uint3 dim = ax_dim;
  float3 dir = ax_wray_dir;
  printf("intersected random: %f %f %f\n", dir.x, dir.y, dir.z);
}

extern "C" ax_kernel void __closesthit__minimum_intersect() { printf("intersected mini\n"); }

extern "C" ax_kernel void __exception__exception_handler() {
  unsigned int type = optixGetExceptionCode();
  printf("Optix device exception : code=%u\n", type);
}

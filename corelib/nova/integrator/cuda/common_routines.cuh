#ifndef COMMON_ROUTINES_CUH
#define COMMON_ROUTINES_CUH

#include "camera/nova_camera.h"
#include "device_internal.h"
#include "gpu/nova_gpu.h"
#include "ray/Ray.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/TextureContext.h"
#include <internal/common/math/math_camera.h>
#include <internal/common/math/math_texturing.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>

// TODO: Share functions with CPU integrators

ax_device_force_inlined void shade(render_buffer_s &render_buffer, float u, float v, const glm::vec4 &color, unsigned channels = 4) {
  unsigned px = math::texture::uvToPixel(u, render_buffer.width - 1);
  unsigned py = math::texture::uvToPixel(v, render_buffer.height - 1);
  unsigned offset = (py * render_buffer.width + px) * channels;
  render_buffer.render_target[offset] = color.r;
  render_buffer.render_target[offset + 1] = color.g;
  render_buffer.render_target[offset + 2] = color.b;
  render_buffer.render_target[offset + 3] = color.a;
}
ax_device_force_inlined void shade(render_buffer_s &render_buffer, unsigned x, unsigned y, const glm::vec4 &color, unsigned channels = 4) {
  unsigned offset = (y * render_buffer.width + x) * channels;
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

ax_device_only nova::Ray compute_screen_ray(float ndc_x, float ndc_y, const nova::camera::CameraResourcesHolder &camera) {
  math::camera::camera_ray c_ray = math::camera::ray_inv_mat(ndc_x, ndc_y, camera.inv_P, camera.inv_V);
  return {c_ray.near, c_ray.far};
}

ax_device_only glm::vec4 sample_envmap(const nova::Ray &ray, const nova::device_traversal_param_s *params, bool use_interop = true) {
  nova::texturing::texture_data_aggregate_s data = {};
  nova::texturing::TextureCtx texture_ctx(params->texture_bundle_views, use_interop);
  data.geometric_data.sampling_vector = ray.direction;
  data.texture_ctx = &texture_ctx;
  nova::texturing::EnvmapTexture img(params->current_envmap_index);
  return img.sample(0, 0, data);
}

#endif

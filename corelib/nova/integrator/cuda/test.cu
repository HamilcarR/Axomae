#include "device_internal.h"
#include "gpu/nova_gpu.h"
#include "integrator/gpu_launcher.h"
#include "texturing/NovaTextureInterface.h"
#include "texturing/TextureContext.h"
#include <internal/common/math/math_camera.h>
#include <internal/common/math/math_texturing.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>

namespace nova {

  ax_device_force_inlined void shade(render_buffer_s &render_buffer, float u, float v, const glm::vec4 &color) {
    unsigned px = math::texture::uvToPixel(u, render_buffer.width - 1);
    unsigned py = math::texture::uvToPixel(v, render_buffer.height - 1);
    unsigned offset = (py * render_buffer.width + px) * 4;
    render_buffer.render_target[offset] = color.r;
    render_buffer.render_target[offset + 1] = color.g;
    render_buffer.render_target[offset + 2] = color.b;
    render_buffer.render_target[offset + 3] = color.a;
  }

  ax_kernel void test_envmap(device_traversal_param_s *params) {

    float u = (float)ax_device_thread_idx_x / (float)params->width;
    float v = (float)ax_device_thread_idx_y / (float)params->height;
    if (u > 1.f || v > 1.f)
      return;
    render_buffer_s rbuffer;
    rbuffer.width = params->width;
    rbuffer.height = params->height;
    rbuffer.render_target = params->render_buffers.partial_buffer;
    nova::texturing::texture_data_aggregate_s data = {};
    nova::texturing::TextureCtx texture_ctx(params->texture_bundle_views, false);
    data.texture_ctx = &texture_ctx;
    nova::texturing::ImageTexture<float> img(2);
    glm::vec4 pixel = img.sample(u, v, data);

    shade(rbuffer, u, v, pixel);
  }

  void device_test_integrator(const device_traversal_param_s &traversal_parameters, nova_eng_internals &nova_internals) {
    dim3 blocks{100, 100, 1};
    dim3 threads{32, 32, 1};
    auto err = device::gpgpu::allocate_buffer(sizeof(device_traversal_param_s));
    DEVICE_ERROR_CHECK(err.error_status);
    DEVICE_ERROR_CHECK(
        device::gpgpu::copy_buffer(&traversal_parameters, err.device_ptr, sizeof(device_traversal_param_s), device::gpgpu::HOST_DEVICE).error_status);

    test_envmap<<<blocks, threads>>>(static_cast<device_traversal_param_s *>(err.device_ptr));
  }
}  // namespace nova

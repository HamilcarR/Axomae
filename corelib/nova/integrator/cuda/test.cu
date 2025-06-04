#include "common_routines.cuh"

#include "integrator/gpu_launcher.h"
namespace nova {

  ax_kernel void gen_ray(const device_traversal_param_s *params) {
    float u = (float)ax_device_thread_idx_x / (float)params->width;
    float v = (float)ax_device_thread_idx_y / (float)params->height;
    if (u > 1.f || v > 1.f)
      return;
    float ndc_x = (u * 2.f) - 1.f;
    float ndc_y = (v * 2.f) - 1.f;
    auto ray = compute_screen_ray(ndc_x, ndc_y, params->camera);
    glm::vec4 pixel = sample_envmap(ray, params);
    render_buffer_s rbuffer;
    rbuffer.width = params->width;
    rbuffer.height = params->height;
    rbuffer.render_target = params->render_buffers.partial_buffer;
    shade(rbuffer, u, v, pixel);
  }

  void device_test_integrator(const device_traversal_param_s &traversal_parameters, nova_eng_internals &nova_internals) {
    dim3 blocks{100, 100, 1};
    dim3 threads{32, 32, 1};
    auto err = device::gpgpu::allocate_buffer(sizeof(device_traversal_param_s));
    DEVICE_ERROR_CHECK(err.error_status);
    DEVICE_ERROR_CHECK(
        device::gpgpu::copy_buffer(&traversal_parameters, err.device_ptr, sizeof(device_traversal_param_s), device::gpgpu::HOST_DEVICE).error_status);

    gen_ray<<<blocks, threads>>>(static_cast<const device_traversal_param_s *>(err.device_ptr));
    device::gpgpu::synchronize_device();
    DEVICE_ERROR_CHECK(device::gpgpu::deallocate_buffer(err.device_ptr).error_status);
  }
}  // namespace nova

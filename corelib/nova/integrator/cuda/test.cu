#include "../gpu_launcher.h"
#include "gpu/nova_gpu.h"
#include <internal/device/gpgpu/device_utils.h>
#include <internal/geometry/Object3D.h>
namespace nova {

  ax_kernel void test(const device_traversal_param_s *params) {
    auto objs = params->mesh_bundle_views.getTriangleGeometryViews();
    if (ax_device_linearRM3D_idx < objs[0].uv.size()) {
      auto uv = objs[0].uv;
      printf("%f\n", uv[ax_device_linearRM3D_idx]);
    }

    return;
  }

  void device_test_integrator(const device_traversal_param_s &traversal_parameters, nova_eng_internals &nova_internals) {
    void *d_traversal_params = nullptr;
    DEVICE_ERROR_CHECK(cudaMalloc(&d_traversal_params, sizeof(device_traversal_param_s)));
    DEVICE_ERROR_CHECK(cudaMemcpy(d_traversal_params, &traversal_parameters, sizeof(device_traversal_param_s), cudaMemcpyHostToDevice));
    test<<<dim3(traversal_parameters.width, traversal_parameters.height, 1), dim3(32, 32, 1)>>>(
        static_cast<const device_traversal_param_s *>(d_traversal_params));
    DEVICE_ERROR_CHECK(cudaDeviceSynchronize());
    DEVICE_ERROR_CHECK(cudaFree(d_traversal_params));
  }

}  // namespace nova

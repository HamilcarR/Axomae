#include "nova_shape.h"
#include <internal/device/gpgpu/device_transfer_interface.h>

namespace nova::shape {

  void ShapeResourcesHolder::init() {
    triangle_meshes_view = axstd::span(triangle_meshes.data(), triangle_meshes.size());
#ifdef AXOMAE_USE_CUDA
    device::gpgpu::pin_host_memory(
        triangle_meshes_view.data(), triangle_meshes_view.size() * sizeof(const Object3D *), device::gpgpu::PIN_MODE_DEFAULT);
    is_mesh_structure_pinned = true;
#endif
    Triangle::init(&triangle_meshes_view);
  }

  void ShapeResourcesHolder::release() {
#ifdef AXOMAE_USE_CUDA
    if (is_mesh_structure_pinned)
      device::gpgpu::unpin_host_memory(triangle_meshes_view.data());
#endif
  }

}  // namespace nova::shape
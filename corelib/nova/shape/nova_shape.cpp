#include "nova_shape.h"
#include <internal/device/gpgpu/device_transfer_interface.h>

namespace nova::shape {

  void ShapeResourcesHolder::init() {
    triangle_mesh_storage.init();
    updateMeshBuffers();
  }
  void ShapeResourcesHolder::updateMeshBuffers() {
    Triangle::updateCpuMeshList(&triangle_mesh_storage.getCPUBuffersView());
#ifdef AXOMAE_USE_CUDA
    triangle_mesh_storage.mapBuffers();
    Triangle::updateGpuMeshList(&triangle_mesh_storage.getGPUBuffersView());
#endif
  }

  void ShapeResourcesHolder::addTriangleMeshGPU(const triangle::mesh_vbo_ids &mesh_vbos) { triangle_mesh_storage.addGeometryGPU(mesh_vbos); }

  void ShapeResourcesHolder::release() { triangle_mesh_storage.release(); }

}  // namespace nova::shape
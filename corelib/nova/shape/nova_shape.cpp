#include "nova_shape.h"
#include <internal/device/gpgpu/device_transfer_interface.h>

namespace nova::shape {

  void ShapeResourcesHolder::init(const shape_init_record_t &startup_data) { mesh_transform_storage.init(startup_data.total_triangle_meshes); }

  void ShapeResourcesHolder::lockResources() {
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

  void ShapeResourcesHolder::releaseResources() { triangle_mesh_storage.release(); }

}  // namespace nova::shape
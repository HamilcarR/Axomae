#include "nova_shape.h"
#include "shape_datastructures.h"
#include <internal/device/gpgpu/device_transfer_interface.h>

namespace nova::shape {

  void ShapeResourcesHolder::init(const shape_init_record_t &startup_data) {
    transform_storage.init(startup_data.total_triangle_meshes);
    storage.allocTriangles(startup_data.total_triangles);
  }

  void ShapeResourcesHolder::lockResources() {
    triangle_mesh_storage.mapResrc();
    mapBuffers();
  }

  void ShapeResourcesHolder::updateSharedBuffers() {
    transform::mesh_transform_views_t transform_views = transform_storage.getTransformViews();
    triangle::mesh_vertex_attrib_views_t geometry_triangle_views = triangle_mesh_storage.getGeometryViews();
    shared_buffers.set(transform_views, geometry_triangle_views);
  }

  void ShapeResourcesHolder::mapBuffers() {
    updateSharedBuffers();
#ifdef AXOMAE_USE_CUDA
    triangle_mesh_storage.mapBuffers();
#endif
  }

  MeshBundleViews ShapeResourcesHolder::getMeshSharedViews() const {
    MeshBundleViews shared_views = MeshBundleViews(transform_storage.getTransformViews(), triangle_mesh_storage.getGeometryViews());
    return shared_views;
  }

  void ShapeResourcesHolder::addTriangleMesh(const triangle::mesh_vbo_ids &mesh_vbos) { triangle_mesh_storage.addGeometry(mesh_vbos); }

  void ShapeResourcesHolder::addTriangleMesh(Object3D triangle_mesh, const glm::mat4 &transform) {
    std::size_t mesh_index = triangle_mesh_storage.addGeometry(triangle_mesh);
    transform_storage.add(transform, mesh_index);
  }
  void ShapeResourcesHolder::releaseResources() { triangle_mesh_storage.release(); }

}  // namespace nova::shape

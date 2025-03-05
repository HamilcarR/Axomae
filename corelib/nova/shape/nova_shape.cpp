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
    shared_buffers.geometry = geometry_triangle_views;
    shared_buffers.transforms = transform_views;
  }

  void ShapeResourcesHolder::mapBuffers() {
    updateSharedBuffers();
#ifdef AXOMAE_USE_CUDA
    triangle_mesh_storage.mapBuffers();
#endif
  }

  mesh_shared_views_t ShapeResourcesHolder::getMeshSharedViews() const {
    mesh_shared_views_t shared_views;
    shared_views.transforms = transform_storage.getTransformViews();
    shared_views.geometry = triangle_mesh_storage.getGeometryViews();
    return shared_views;
  }

  void ShapeResourcesHolder::addTriangleMesh(const triangle::mesh_vbo_ids &mesh_vbos) { triangle_mesh_storage.addGeometry(mesh_vbos); }

  void ShapeResourcesHolder::addTriangleMesh(Object3D triangle_mesh, const glm::mat4 &transform) {
    std::size_t mesh_index = triangle_mesh_storage.addGeometry(triangle_mesh);
    transform_storage.add(transform, mesh_index);
  }
  void ShapeResourcesHolder::releaseResources() { triangle_mesh_storage.release(); }

}  // namespace nova::shape

#include "nova_shape.h"
#include "shape_datastructures.h"
#include <internal/device/gpgpu/device_transfer_interface.h>

namespace nova::shape {

  /************************************************************************************************************************/

  void MeshStorageIndexer::allocate(const shape_init_record_t &shape_infos) {
    triangle_mesh_storage.allocate(shape_infos.total_triangle_meshes);
    TRIANGLE_MESH_PADDING = triangle_mesh_storage.size();
  }

  void MeshStorageIndexer::clear() { triangle_mesh_storage.clear(); }

  std::size_t MeshStorageIndexer::addTriangleMesh(const triangle::mesh_vbo_ids &vbos) { return triangle_mesh_storage.addGeometry(vbos); }
  std::size_t MeshStorageIndexer::addTriangleMesh(const Object3D &geometry) { return triangle_mesh_storage.addGeometry(geometry); }
  void MeshStorageIndexer::release() { triangle_mesh_storage.release(); }
  void MeshStorageIndexer::mapBuffers() { triangle_mesh_storage.mapBuffers(); }
  void MeshStorageIndexer::mapResources() { triangle_mesh_storage.mapResrc(); }

  /************************************************************************************************************************/
  void ShapeResourcesHolder::init(const shape_init_record_t &startup_data) {
    transform_storage.init(startup_data.total_triangle_meshes);
    mesh_indexer.allocate(startup_data);
    storage.allocTriangles(startup_data.total_triangles);
  }

  void ShapeResourcesHolder::lockResources() {
    mesh_indexer.mapResources();
    mapBuffers();
  }

  void ShapeResourcesHolder::mapBuffers() { mesh_indexer.mapBuffers(); }

  MeshBundleViews ShapeResourcesHolder::getMeshSharedViews() const {
    MeshBundleViews shared_views = MeshBundleViews(transform_storage.getTransformViews(), mesh_indexer.getTriangleMeshViews());
    return shared_views;
  }

  void ShapeResourcesHolder::addTransform(const glm::mat4 &transform, std::size_t mesh_index) { transform_storage.add(transform, mesh_index); }

  void ShapeResourcesHolder::releaseResources() { mesh_indexer.release(); }

}  // namespace nova::shape

#ifndef GEOMETRY_H
#define GEOMETRY_H
#include <shape/nova_shape.h>

class MeshContextBuilder {
  nova::shape::transform::TransformStorage transform_storage;
  nova::shape::triangle::GeometryReferenceStorage host_storage;

 public:
  MeshContextBuilder(std::size_t allocate_num_meshes) {
    host_storage.allocate(allocate_num_meshes);
    transform_storage.allocate(allocate_num_meshes);
  }

  void addMesh(const Object3D &obj, const glm::mat4 &transform) {
    std::size_t index = host_storage.addGeometry(obj);
    transform_storage.add(transform, index);
  }

  nova::shape::MeshCtx getCtx() const {
    nova::shape::MeshBundleViews bundle = nova::shape::MeshBundleViews(transform_storage.getTransformViews(), host_storage.getGeometryViews());
    return nova::shape::MeshCtx(bundle);
  }
};

#endif  // GEOMETRY_H

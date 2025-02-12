#ifndef NOVA_SHAPE_H
#define NOVA_SHAPE_H
#include "ShapeInterface.h"
#include "internal/memory/MemoryArena.h"
#include "mesh_transform_storage.h"
#include "shape/Triangle.h"
#include "shape/shape_datastructures.h"
#include "triangle_mesh_storage.h"

namespace nova {
  class Ray;
}

namespace nova::shape {

  struct shape_init_record_t {
    std::size_t total_triangle_meshes;
  };

  class ShapeResourcesHolder {
    /* Generic shape pointers*/
    std::vector<NovaShapeInterface> shapes;
    triangle::Storage triangle_mesh_storage;
    transform::Storage mesh_transform_storage;
    mesh_shared_views_t shared_buffers;

   public:
    CLASS_M(ShapeResourcesHolder)

    /* Since the creation of shapes is incremental (we don't copy a whole array here) , provide the current offset of the object to create.*/
    template<class T, class... Args>
    NovaShapeInterface add_shape(T *allocation_buffer, std::size_t offset, Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Shape type.");
      T *allocated_ptr = core::memory::MemoryArena<>::construct<T>(&allocation_buffer[offset], std::forward<Args>(args)...);
      shapes.push_back(allocated_ptr);
      AX_ASSERT_NOTNULL(shapes.back().get());
      return shapes.back();
    }

    void addTriangleMesh(Object3D triangle_mesh, const glm::mat4 &transform);
    void addTriangleMeshGPU(const triangle::mesh_vbo_ids &mesh_vbos);
    std::vector<NovaShapeInterface> &get_shapes() { return shapes; }
    ax_no_discard const std::vector<NovaShapeInterface> &get_shapes() const { return shapes; }

    void clear() {
      shapes.clear();
      triangle_mesh_storage.clear();
      mesh_transform_storage.clear();
    }

    void updateSharedBuffers();
    void init(const shape_init_record_t &init_data);
    void lockResources();
    void releaseResources();
    void mapBuffers();
    const triangle::Storage &getTriangleMeshStorage() const { return triangle_mesh_storage; }
    mesh_shared_views_t getMeshSharedViews() const;
  };

}  // namespace nova::shape

#endif  // NOVA_SHAPE_H

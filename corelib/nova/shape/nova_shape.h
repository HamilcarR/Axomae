#ifndef NOVA_SHAPE_H
#define NOVA_SHAPE_H
#include "ShapeInterface.h"
#include "internal/memory/MemoryArena.h"
#include <memory>

namespace nova {
  class Ray;
}
namespace nova::shape {

  class ShapeResourcesHolder {

   private:
    // TODO: Use allocator ?
    std::vector<NovaShapeInterface> shapes;
    std::vector<const Object3D *> triangle_meshes;

    axstd::span<const Object3D *>
        triangle_meshes_view;  // Gets initialized from triangle_meshes , and initializes the triangle mesh list of the Triangle class
    bool is_mesh_structure_pinned;

   public:
    CLASS_CM(ShapeResourcesHolder)

    /* Since the creation of shapes is incremental (we don't copy a whole array here) , provide the current offset of the object to create.*/
    template<class T, class... Args>
    NovaShapeInterface add_shape(T *allocation_buffer, std::size_t offset, Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Shape type.");
      T *allocated_ptr = core::memory::MemoryArena<>::construct<T>(&allocation_buffer[offset], std::forward<Args>(args)...);
      shapes.push_back(allocated_ptr);
      AX_ASSERT_NOTNULL(shapes.back().get());
      return shapes.back();
    }

    void addTriangleMesh(const Object3D *triangle_mesh) { triangle_meshes.push_back(triangle_mesh); }
    std::vector<NovaShapeInterface> &get_shapes() { return shapes; }
    ax_no_discard const std::vector<NovaShapeInterface> &get_shapes() const { return shapes; }

    void clear() {
      shapes.clear();
      triangle_meshes.clear();
    }

    void init();
    void release();
  };

}  // namespace nova::shape

#endif  // NOVA_SHAPE_H

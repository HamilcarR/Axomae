#ifndef NOVA_SHAPE_H
#define NOVA_SHAPE_H
#include "ShapeInterface.h"
#include "mesh_transform_storage.h"
#include "shape/Triangle.h"
#include "shape/shape_datastructures.h"
#include "triangle_mesh_storage.h"
#include <internal/common/axstd/managed_buffer.h>
#include <internal/memory/MemoryArena.h>

namespace nova {
  class Ray;
}

namespace nova::shape {

  struct shape_init_record_t {
    std::size_t total_triangle_meshes;
    std::size_t total_triangles;
  };

  /* Make sure to pre allocate the shapes structures beforehand to avoid vector reallocation and dangling pointer.*/
  class ShapeStorage {
    axstd::managed_vector<NovaShapeInterface> shapes{};
    axstd::managed_vector<Triangle> triangles{};
    axstd::managed_vector<Sphere> spheres{};
    axstd::managed_vector<Box> boxes{};
    axstd::managed_vector<Square> squares{};

   public:
    NovaShapeInterface add(const Triangle &triangle) { return appendShape(triangle, triangles); }
    NovaShapeInterface add(const Sphere &sphere) { return appendShape(sphere, spheres); }
    NovaShapeInterface add(const Box &box) { return appendShape(box, boxes); }
    NovaShapeInterface add(const Square &square) { return appendShape(square, squares); }

    void allocTriangles(std::size_t total_triangle_shapes) { triangles.reserve(total_triangle_shapes); }
    void allocSpheres(std::size_t total_spheres) { spheres.reserve(total_spheres); }
    void allocBoxes(std::size_t total_boxes) { boxes.reserve(total_boxes); }
    void allocSquares(std::size_t total_squares) { squares.reserve(total_squares); }

    void clear() {
      shapes.clear();
      triangles.clear();
      spheres.clear();
      boxes.clear();
      squares.clear();
    }

   private:
    template<class T>
    NovaShapeInterface appendShape(const T &shape, axstd::managed_vector<T> &shape_vector) {
      AX_ASSERT_GT(shape_vector.capacity(), 0);
      AX_ASSERT_LE(shape_vector.size(), shape_vector.capacity());

      shape_vector.push_back(shape);
      NovaShapeInterface shape_ptr = &shape_vector.back();
      shapes.push_back(shape_ptr);
      return shape_ptr;
    }
  };

  class ShapeResourcesHolder {
    /* Generic shape pointers*/
    ShapeStorage storage;
    MeshBundleViews shared_buffers;
    triangle::Storage triangle_mesh_storage;
    transform::Storage transform_storage;

   public:
    CLASS_M(ShapeResourcesHolder)

    template<class T, class... Args>
    NovaShapeInterface addShape(Args &&...args) {
      static_assert(core::has<T, TYPELIST>::has_type, "Provided type is not a Shape type.");
      const T tshape = T(std::forward<Args>(args)...);
      return storage.add(tshape);
    }

    void addTriangleMesh(Object3D triangle_mesh, const glm::mat4 &transform);
    void addTriangleMesh(const triangle::mesh_vbo_ids &mesh_vbos);

    void clear() {
      storage.clear();
      triangle_mesh_storage.clear();
      transform_storage.clear();
    }

    void updateSharedBuffers();
    void init(const shape_init_record_t &init_data);
    void lockResources();
    void releaseResources();
    void mapBuffers();
    const triangle::Storage &getTriangleMeshStorage() const { return triangle_mesh_storage; }
    MeshBundleViews getMeshSharedViews() const;
  };

}  // namespace nova::shape

#endif  // NOVA_SHAPE_H

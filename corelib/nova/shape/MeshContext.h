#ifndef MESHCONTEXT_H
#define MESHCONTEXT_H
#include "shape_datastructures.h"
#include <internal/device/gpgpu/device_utils.h>

namespace nova::shape {

  class MeshBundleViews {
    transform::mesh_transform_views_t transforms{};
    triangle::mesh_vertex_attrib_views_t triangle_mesh_geometry{};
    /* To add more shapes here we can :
     * 1) Implement a new storage class in that shape's namespace. (Will need a little refactor for cache optimization first)
     * 2) Retrieve its views on its geometry representation.
     * 3) Add as its mesh_index a padding equal to the previous structure's number of primitives.
     */

   public:
    CLASS_DCM(MeshBundleViews)

    ax_device_callable MeshBundleViews(const transform::mesh_transform_views_t &t_view, const triangle::mesh_vertex_attrib_views_t &geom)
        : transforms(t_view), triangle_mesh_geometry(geom) {}

    ax_device_callable void set(const transform::mesh_transform_views_t &t, const triangle::mesh_vertex_attrib_views_t &g) {
      transforms = t;
      triangle_mesh_geometry = g;
    }

    ax_device_callable const axstd::span<const Object3D> &getTriangleGeometryViews() const {
#ifdef __CUDA_ARCH__
      return triangle_mesh_geometry.device_geometry_view;
#else
      return triangle_mesh_geometry.host_geometry_view;
#endif
    }
    ax_device_callable std::size_t triangleCount(size_t mesh_index) const {
      auto triangle_views = getTriangleGeometryViews();
      AX_ASSERT(!triangle_views.empty(), "");
      AX_ASSERT_LT(mesh_index, triangle_views.size());
      const Object3D &obj = triangle_views[mesh_index];
      AX_ASSERT(!obj.indices.empty(), "");
      return obj.indices.size() / 3;
    }

    ax_device_callable std::size_t triMeshCount() const {
      auto triangle_views = getTriangleGeometryViews();
      AX_ASSERT(!triangle_views.empty(), "");
      return triangle_views.size();
    }

    ax_device_callable const Object3D &getTriangleMesh(size_t mesh_id) const {
      auto triangle_views = getTriangleGeometryViews();
      AX_ASSERT_LT(mesh_id, triangle_views.size());
      return triangle_views[mesh_id];
    }

    ax_device_callable_inlined transform4x4_t reconstructTransform4x4(size_t mesh_index) const {
      AX_ASSERT_LT(mesh_index, transforms.mesh_offsets_to_matrix.size());
      std::size_t transform_offset = transform::get_transform_offset(mesh_index, transforms);
      AX_ASSERT_NEQ(transform_offset, transform::INVALID_OFFSET);
      transform4x4_t returned_transform4x4{};
      int err = transform::reconstruct_transform4x4(returned_transform4x4, transform_offset, transforms);
      AX_ASSERT(err == 0, "Error reconstructing transformation matrix");
      return returned_transform4x4;
    }

    ax_device_callable size_t getTransformOffset(size_t mesh_index) const { return transform::get_transform_offset(mesh_index, transforms); }
  };

  /**
   * @brief: Interface containing references to geometry buffers.
   * Passed to intersection routines to retrieve mesh and transformation data.
   */
  class MeshCtx {
    MeshBundleViews geometry_views;

   public:
    CLASS_DCM(MeshCtx)

    ax_device_callable MeshCtx(const MeshBundleViews &geo) : geometry_views(geo) {}

    /**
     * @brief: Retrieves a triangle based mesh from a mesh_index.
     */
    ax_device_callable const Object3D &getTriMesh(size_t index) const { return geometry_views.getTriangleMesh(index); }

    /**
     * @brief: Retrieves the transformation of a triangle based mesh.
     */
    ax_device_callable_inlined transform4x4_t getTriMeshTransform(size_t mesh_index) const {
      return geometry_views.reconstructTransform4x4(mesh_index);
    }

    ax_device_callable const MeshBundleViews &getGeometryViews() const { return geometry_views; }
  };

}  // namespace nova::shape

#endif

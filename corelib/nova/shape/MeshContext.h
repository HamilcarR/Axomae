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

    ax_device_callable MeshBundleViews(const transform::mesh_transform_views_t &transform_views,
                                       const triangle::mesh_vertex_attrib_views_t &geometry);

    ax_device_callable void set(const transform::mesh_transform_views_t &transforms, const triangle::mesh_vertex_attrib_views_t &geometry);

    /**
     * See comment for reconstruct_transform4x4()
     */
    ax_device_callable transform::transform4x4_t reconstructTransform4x4(std::size_t mesh_index) const;
    ax_device_callable std::size_t getTransformOffset(std::size_t mesh_index) const;

    ax_device_callable const axstd::span<const Object3D> &getTriangleGeometryViews() const {
#ifdef __CUDA_ARCH__
      return triangle_mesh_geometry.device_geometry_view;
#else
      return triangle_mesh_geometry.host_geometry_view;
#endif
    }

    ax_device_callable const transform::mesh_transform_views_t &getTransforms() const { return transforms; }
    ax_device_callable const Object3D &getTriangleMesh(std::size_t mesh_id) const;
    ax_device_callable std::size_t triMeshCount() const;
    ax_device_callable std::size_t triangleCount(std::size_t mesh_index) const;
  };

  /**
   * @brief: Interface containing references to geometry buffers.
   * Passed to intersection routines to retrieve mesh and transformation data.
   */
  class MeshCtx {
    MeshBundleViews geometry_views;

   public:
    CLASS_DCM(MeshCtx)

    ax_device_callable MeshCtx(const MeshBundleViews &geometry);

    /**
     * @brief: Retrieves a triangle based mesh from a mesh_index.
     */
    ax_device_callable const Object3D &getTriMesh(std::size_t mesh_index) const;

    /**
     * @brief: Retrieves the transformation of a triangle based mesh.
     */
    ax_device_callable transform::transform4x4_t getTriMeshTransform(std::size_t mesh_index) const;

    ax_device_callable const MeshBundleViews &getGeometryViews() const { return geometry_views; }
  };

}  // namespace nova::shape

#endif

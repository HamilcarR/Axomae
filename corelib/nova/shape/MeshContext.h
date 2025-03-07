#ifndef MESHCONTEXT_H
#define MESHCONTEXT_H
#include "shape_datastructures.h"
#include <internal/device/gpgpu/device_utils.h>

namespace nova::shape {

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

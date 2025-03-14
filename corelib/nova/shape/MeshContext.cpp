#include "MeshContext.h"
#include "shape_datastructures.h"
#include <internal/common/axstd/span.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>

namespace nova::shape {

  ax_device_callable MeshBundleViews::MeshBundleViews(const transform::mesh_transform_views_t &t_view,
                                                      const triangle::mesh_vertex_attrib_views_t &geom)
      : transforms(t_view), triangle_mesh_geometry(geom) {}

  ax_device_callable void MeshBundleViews::set(const transform::mesh_transform_views_t &t, const triangle::mesh_vertex_attrib_views_t &g) {
    transforms = t;
    triangle_mesh_geometry = g;
  }

  ax_device_callable std::size_t MeshBundleViews::triangleCount(std::size_t mesh_index) const {
    auto triangle_views = getTriangleGeometryViews();
    AX_ASSERT(!triangle_views.empty(), "");
    AX_ASSERT_LT(mesh_index, triangle_views.size());
    const Object3D &obj = triangle_views[mesh_index];
    AX_ASSERT(!obj.indices.empty(), "");
    return obj.indices.size() / 3;
  }

  ax_device_callable std::size_t MeshBundleViews::triMeshCount() const {
    auto triangle_views = getTriangleGeometryViews();
    AX_ASSERT(!triangle_views.empty(), "");
    return triangle_views.size();
  }

  ax_device_callable const axstd::span<const Object3D> &MeshBundleViews::getTriangleGeometryViews() const {
#ifdef __CUDA_ARCH__
    return triangle_mesh_geometry.device_geometry_view;
#else
    return triangle_mesh_geometry.host_geometry_view;
#endif
  }

  ax_device_callable const Object3D &MeshBundleViews::getTriangleMesh(std::size_t mesh_id) const {
    auto triangle_views = getTriangleGeometryViews();
    AX_ASSERT_LT(mesh_id, triangle_views.size());
    return triangle_views[mesh_id];
  }

  ax_device_callable transform::transform4x4_t MeshBundleViews::reconstructTransform4x4(std::size_t mesh_index) const {
    AX_ASSERT_LT(mesh_index, transforms.mesh_offsets_to_matrix.size());
    std::size_t transform_offset = transform::get_transform_offset(mesh_index, transforms);
    AX_ASSERT_NEQ(transform_offset, transform::INVALID_OFFSET);
    transform::transform4x4_t returned_transform4x4{};
    int err = transform::reconstruct_transform4x4(returned_transform4x4, transform_offset, transforms);
    AX_ASSERT(err == 0, "Error reconstructing transformation matrix");
    return returned_transform4x4;
  }

  ax_device_callable std::size_t MeshBundleViews::getTransformOffset(std::size_t mesh_index) const {
    return transform::get_transform_offset(mesh_index, transforms);
  }
  /**************************************************************************************************************************/

  ax_device_callable MeshCtx::MeshCtx(const MeshBundleViews &geo) : geometry_views(geo) {}

  ax_device_callable const Object3D &MeshCtx::getTriMesh(std::size_t index) const { return geometry_views.getTriangleMesh(index); }

  ax_device_callable transform::transform4x4_t MeshCtx::getTriMeshTransform(std::size_t mesh_index) const {
    return geometry_views.reconstructTransform4x4(mesh_index);
  }

}  // namespace nova::shape

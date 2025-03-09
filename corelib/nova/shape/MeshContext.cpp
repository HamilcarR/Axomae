#include "MeshContext.h"
#include "shape_datastructures.h"
#include <internal/common/axstd/span.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>

namespace nova::shape {

  ax_device_callable MeshBundleViews::MeshBundleViews(const transform::mesh_transform_views_t &t_view,
                                                      const triangle::mesh_vertex_attrib_views_t &geom)
      : transforms(t_view), geometry(geom) {}

  void MeshBundleViews::set(const transform::mesh_transform_views_t &t, const triangle::mesh_vertex_attrib_views_t &g) {
    transforms = t;
    geometry = g;
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

  ax_device_callable_inlined const Object3D &get_triangle_mesh(std::size_t index, const MeshBundleViews &geometry_views) {
#ifndef __CUDA_ARCH__
    const auto &mesh_array = geometry_views.getTriangleGeometryViews().host_geometry_view;
#else
    const auto &mesh_array = geometry_views.getTriangleGeometryViews().device_geometry_view;
#endif
    AX_ASSERT_LT(index, mesh_array.size());
    return mesh_array[index];
  }

  ax_device_callable MeshCtx::MeshCtx(const MeshBundleViews &geo) : geometry_views(geo) {}

  ax_device_callable const Object3D &MeshCtx::getTriMesh(std::size_t index) const { return get_triangle_mesh(index, geometry_views); }

  ax_device_callable transform::transform4x4_t MeshCtx::getTriMeshTransform(std::size_t mesh_index) const {
    return geometry_views.reconstructTransform4x4(mesh_index);
  }

}  // namespace nova::shape

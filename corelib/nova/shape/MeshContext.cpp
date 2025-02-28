#include "MeshContext.h"
#include "shape_datastructures.h"
#include <internal/common/axstd/span.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/macro/project_macros.h>

namespace nova::shape {

  ax_device_callable_inlined const Object3D &get_mesh(std::size_t index, const mesh_shared_views_t &geometry_views) {
#ifndef __CUDA_ARCH__
    const auto &mesh_array = geometry_views.geometry.host_geometry_view;
#else
    const auto &mesh_array = geometry_views.geometry.device_geometry_view;  // TODO: Change after implementing managed vectors
#endif
    AX_ASSERT_LT(index, mesh_array.size());
    return mesh_array[index];
  }

  ax_device_callable MeshCtx::MeshCtx(const mesh_shared_views_t &geo) : geometry_views(geo) {}

  ax_device_callable const Object3D &MeshCtx::getTriMesh(std::size_t index) const { return get_mesh(index, geometry_views); }

  ax_device_callable transform::transform4x4_t MeshCtx::getTriMeshTransform(std::size_t mesh_index) const {
    AX_ASSERT_LT(mesh_index, geometry_views.transforms.mesh_offsets_to_matrix.size());
    std::size_t matrix_offset = geometry_views.transforms.mesh_offsets_to_matrix[mesh_index];
    AX_ASSERT_LT(matrix_offset, geometry_views.transforms.matrix_components_view.size());
    transform::transform4x4_t transform{};
    int err = transform::reconstruct_transform4x4(transform, matrix_offset, geometry_views.transforms);
    AX_ASSERT(err == 0, "Error reconstructing transformation matrix for mesh");
    return transform;
  }

}  // namespace nova::shape

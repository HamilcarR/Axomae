#ifndef TRIANGLE_MESH_STORAGE_H
#define TRIANGLE_MESH_STORAGE_H
#include "interfaces/DeviceReferenceStorageInterface.h"
#include "shape/shape_datastructures.h"
#include <internal/common/axstd/managed_buffer.h>
#include <internal/geometry/Object3D.h>

#ifdef AXOMAE_USE_CUDA
#  include "gpu/mesh_device_resources.h"
#endif
using ConstIdxMeshesView = axstd::span<const Object3D>;
using IdxMeshesView = axstd::span<Object3D>;

namespace nova::gpu {
  template<class T>
  class DeviceBufferTracker;
}

namespace nova::shape::triangle {

  struct mesh_vbo_ids {
    uint32_t vbo_positions;  // TODO: GLuint
    uint32_t vbo_uv;
    uint32_t vbo_tangents;
    uint32_t vbo_normals;
    uint32_t vbo_indices;
  };

  class DummyBufferTracker {
   public:
    DummyBufferTracker() = default;
    DummyBufferTracker(uint32_t, device::gpgpu::ACCESS_TYPE) {}
  };

  template<class T, bool using_gpu = core::build::is_gpu_build>
  using DeviceBuffer = std::conditional_t<using_gpu, gpu::DeviceBufferTracker<T>, DummyBufferTracker>;

  struct mesh_device_buffers {
    DeviceBuffer<float> positions;
    DeviceBuffer<float> uv;
    DeviceBuffer<float> tangents;
    DeviceBuffer<float> normals;
    DeviceBuffer<unsigned> indices;
  };

  struct device_storage {
    std::vector<mesh_device_buffers> buffers_trackers;
    axstd::managed_vector<Object3D> geometry_storage;
    IdxMeshesView geometry_view;
  };

  struct host_storage {
    std::vector<Object3D> geometry_storage;
    IdxMeshesView geometry_view;
  };
  /************************************************************************************************************************/

  class HostPolicy {};
  class DevicePolicy {};

  template<class StoragePolicy = std::conditional_t<core::build::is_gpu_build, DevicePolicy, HostPolicy>>
  class DispatchedGeometryReferenceStorage : public StoragePolicy, public DeviceReferenceStorageInterface {
    host_storage cpu_geometry;
    device_storage gpu_geometry;
    std::size_t container_capacity{};

   public:
    const IdxMeshesView &getCPUBuffersView() const;
    std::size_t size() const override { return container_capacity; }
    void allocate(std::size_t num_meshes);
    void clear() override;
    void mapBuffers() override;
    void mapResources() override;
    void release() override;

    /**
     * @brief Adds a mesh whose geometry representation resides on GPU and is already registered with valid VBOs for each vertex attribute arrays.
     * This is for preventing duplication of geometry data if there's a GL context already working.
     * @returns Returns its index if application is compiled with gpgpu context .
     */
    std::size_t addGeometry(const mesh_vbo_ids &mesh_vbos);

    /**
     *@brief Adds a Mesh whose geometry representation resides on CPU.
     *@returns Returns its index.
     */
    std::size_t addGeometry(const Object3D &geometry);

    const IdxMeshesView &getGPUBuffersView() const;

    mesh_vertex_attrib_views_t getGeometryViews() const;
  };

  using GeometryReferenceStorage = DispatchedGeometryReferenceStorage<>;
}  // namespace nova::shape::triangle
#endif

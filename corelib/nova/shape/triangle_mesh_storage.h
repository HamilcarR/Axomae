#ifndef TRIANGLE_MESH_STORAGE_H
#define TRIANGLE_MESH_STORAGE_H
#include <internal/geometry/Object3D.h>

#ifdef AXOMAE_USE_CUDA
#  include "gpu/mesh_device_resources.h"
#endif
namespace nova::shape::triangle {

  struct mesh_vbo_ids {
#ifdef AXOMAE_USE_CUDA
    uint32_t vbo_positions;
    uint32_t vbo_uv;
    uint32_t vbo_tangents;
    uint32_t vbo_normals;
    uint32_t vbo_indices;
#endif
  };

  struct mesh_device_buffers {
#ifdef AXOMAE_USE_CUDA
    gpu::DeviceBufferTracker<float> positions;
    gpu::DeviceBufferTracker<float> uv;
    gpu::DeviceBufferTracker<float> tangents;
    gpu::DeviceBufferTracker<float> normals;
    gpu::DeviceBufferTracker<unsigned> indices;
#endif
  };

  struct device_storage {
    std::vector<mesh_device_buffers> buffers_trackers;
    std::vector<Object3D> geometry_storage;
    axstd::span<Object3D> geometry_view;
  };

  struct host_storage {
    std::vector<Object3D> geometry_storage;
    axstd::span<Object3D> geometry_view;
  };

  class Storage {
   private:
    host_storage cpu_geometry;
    device_storage gpu_geometry;

   public:
    void addGeometryCPU(const Object3D &geometry);
    void addGeometryGPU(const mesh_vbo_ids &mesh_vbos);
    const axstd::span<Object3D> &getCPUBuffersView() const;
    const axstd::span<Object3D> &getGPUBuffersView() const;
    void clear();
    void mapBuffers();
    void init();
    void release();
  };
}  // namespace nova::shape::triangle
#endif
#ifndef TRIANGLE_MESH_STORAGE_H
#define TRIANGLE_MESH_STORAGE_H
#include "shape/shape_datastructures.h"
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
    /**
     *@brief Adds a Mesh whose geometry representation resides on CPU.
     *@returns Returns its index.
     */
    std::size_t addGeometry(const Object3D &geometry);
    /**
     * @brief Adds a mesh whose geometry representation resides on GPU and is already registered with valid VBOs for each vertex attribute arrays.
     * @returns Returns its index.
     */
    std::size_t addGeometry(const mesh_vbo_ids &mesh_vbos);
    const axstd::span<Object3D> &getCPUBuffersView() const;
    const axstd::span<Object3D> &getGPUBuffersView() const;
    void clear();
    void mapBuffers();
    void init();
    void release();
    mesh_vertex_attrib_views_t getGeometryViews() const;
  };
}  // namespace nova::shape::triangle
#endif

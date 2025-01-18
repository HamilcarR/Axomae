#include "triangle_mesh_storage.h"

namespace nova::shape::triangle {

  void Storage::addGeometryCPU(const Object3D &geometry) { cpu_geometry.geometry_storage.push_back(geometry); }

  void Storage::addGeometryGPU(const mesh_vbo_ids &vbos) {
#ifdef AXOMAE_USE_CUDA
    mesh_device_buffers device_buffers;
    device_buffers.positions = gpu::DeviceBufferTracker<float>(vbos.vbo_positions, device::gpgpu::READ_ONLY);
    device_buffers.uv = gpu::DeviceBufferTracker<float>(vbos.vbo_uv, device::gpgpu::READ_ONLY);
    device_buffers.tangents = gpu::DeviceBufferTracker<float>(vbos.vbo_tangents, device::gpgpu::READ_ONLY);
    device_buffers.normals = gpu::DeviceBufferTracker<float>(vbos.vbo_normals, device::gpgpu::READ_ONLY);
    device_buffers.indices = gpu::DeviceBufferTracker<unsigned>(vbos.vbo_indices, device::gpgpu::READ_ONLY);
    gpu_geometry.buffers_trackers.push_back(std::move(device_buffers));
#endif
  }

#ifdef AXOMAE_USE_CUDA
  inline void map_gpu_resources(mesh_device_buffers &dev_buffers) {
    dev_buffers.positions.mapResource();
    dev_buffers.uv.mapResource();
    dev_buffers.tangents.mapResource();
    dev_buffers.normals.mapResource();
    dev_buffers.indices.mapResource();
  }

  inline void map_buffers(mesh_device_buffers &dev_buffers) {
    dev_buffers.positions.mapBuffer();
    dev_buffers.uv.mapBuffer();
    dev_buffers.tangents.mapBuffer();
    dev_buffers.normals.mapBuffer();
    dev_buffers.indices.mapBuffer();
  }

  static Object3D create_gpu_obj3d(mesh_device_buffers &dev_buffers) {
    Object3D geometry;
    geometry.vertices = dev_buffers.positions.getDeviceBuffer();
    geometry.indices = dev_buffers.indices.getDeviceBuffer();
    geometry.normals = dev_buffers.normals.getDeviceBuffer();
    geometry.tangents = dev_buffers.tangents.getDeviceBuffer();
    geometry.uv = dev_buffers.uv.getDeviceBuffer();
    return geometry;
  }

  inline void unmap_gpu_resources(mesh_device_buffers &dev_buffers) {
    dev_buffers.positions.unmapResource();
    dev_buffers.uv.unmapResource();
    dev_buffers.tangents.unmapResource();
    dev_buffers.normals.unmapResource();
    dev_buffers.indices.unmapResource();
  }
#endif

  void Storage::mapBuffers() {
    cpu_geometry.geometry_view = axstd::span(cpu_geometry.geometry_storage.data(), cpu_geometry.geometry_storage.size());
#ifdef AXOMAE_USE_CUDA
    gpu_geometry.geometry_storage.clear();
    for (auto &tracker : gpu_geometry.buffers_trackers) {
      map_buffers(tracker);
      gpu_geometry.geometry_storage.push_back(create_gpu_obj3d(tracker));
    }
    gpu_geometry.geometry_view = axstd::span(gpu_geometry.geometry_storage.data(), gpu_geometry.geometry_storage.size());
#endif
  }

  void Storage::init() {
#ifdef AXOMAE_USE_CUDA
    for (auto &tracker : gpu_geometry.buffers_trackers) {
      map_gpu_resources(tracker);
    }

#endif
  }

  void Storage::release() {
#ifdef AXOMAE_USE_CUDA
    for (auto &tracker : gpu_geometry.buffers_trackers) {
      unmap_gpu_resources(tracker);
    }
#endif
  }

  const axstd::span<Object3D> &Storage::getCPUBuffersView() const { return cpu_geometry.geometry_view; }

  const axstd::span<Object3D> &Storage::getGPUBuffersView() const { return gpu_geometry.geometry_view; }

  void Storage::clear() {
    cpu_geometry.geometry_storage.clear();
    gpu_geometry.geometry_storage.clear();
    gpu_geometry.buffers_trackers.clear();
  }

}  // namespace nova::shape::triangle
#include "triangle_mesh_storage.h"
#include "shape_datastructures.h"

template<class T>
void map_gpu_resources(T &dev_buffers) {
  if constexpr (core::build::is_gpu_build) {
    dev_buffers.positions.mapResource();
    dev_buffers.uv.mapResource();
    dev_buffers.tangents.mapResource();
    dev_buffers.normals.mapResource();
    dev_buffers.indices.mapResource();
  }
}

template<class T>
void map_buffers(T &dev_buffers) {
  if constexpr (core::build::is_gpu_build) {
    dev_buffers.positions.mapBuffer();
    dev_buffers.uv.mapBuffer();
    dev_buffers.tangents.mapBuffer();
    dev_buffers.normals.mapBuffer();
    dev_buffers.indices.mapBuffer();
  }
}
template<class T>
Object3D create_gpu_obj3d(T &dev_buffers) {
  Object3D geometry{};
  if constexpr (core::build::is_gpu_build) {
    geometry.vertices = dev_buffers.positions.getDeviceBuffer();
    geometry.indices = dev_buffers.indices.getDeviceBuffer();
    geometry.normals = dev_buffers.normals.getDeviceBuffer();
    geometry.tangents = dev_buffers.tangents.getDeviceBuffer();
    geometry.uv = dev_buffers.uv.getDeviceBuffer();
  }
  return geometry;
}

template<class T>
void unmap_gpu_resources(T &dev_buffers) {
  if constexpr (core::build::is_gpu_build) {
    dev_buffers.positions.unmapResource();
    dev_buffers.uv.unmapResource();
    dev_buffers.tangents.unmapResource();
    dev_buffers.normals.unmapResource();
    dev_buffers.indices.unmapResource();
  }
}
namespace nova::shape::triangle {

  /***************************************************************************************************************/
  /* Device compilation path.*/

  template<>
  void DispatchedGeometryReferenceStorage<DevicePolicy>::allocate(std::size_t num_meshes) {
    gpu_geometry.buffers_trackers.reserve(num_meshes);
    cpu_geometry.geometry_storage.reserve(num_meshes);
    container_capacity = num_meshes;
  }

  template<>
  std::size_t DispatchedGeometryReferenceStorage<>::addGeometry(const mesh_vbo_ids &vbos) {
    mesh_device_buffers device_buffers;
    device_buffers.positions = DeviceBuffer<float>(vbos.vbo_positions, device::gpgpu::READ_ONLY);
    device_buffers.uv = DeviceBuffer<float>(vbos.vbo_uv, device::gpgpu::READ_ONLY);
    device_buffers.tangents = DeviceBuffer<float>(vbos.vbo_tangents, device::gpgpu::READ_ONLY);
    device_buffers.normals = DeviceBuffer<float>(vbos.vbo_normals, device::gpgpu::READ_ONLY);
    device_buffers.indices = DeviceBuffer<unsigned>(vbos.vbo_indices, device::gpgpu::READ_ONLY);
    gpu_geometry.buffers_trackers.push_back(std::move(device_buffers));
    return gpu_geometry.buffers_trackers.size() - 1;
  }

  template<>
  const IdxMeshesView &DispatchedGeometryReferenceStorage<DevicePolicy>::getGPUBuffersView() const {
    return gpu_geometry.geometry_view;
  }

  template<>
  void DispatchedGeometryReferenceStorage<DevicePolicy>::mapBuffers() {
    cpu_geometry.geometry_view = axstd::span(cpu_geometry.geometry_storage.data(), cpu_geometry.geometry_storage.size());
    gpu_geometry.geometry_storage.clear();
    for (auto &tracker : gpu_geometry.buffers_trackers) {
      map_buffers(tracker);
      gpu_geometry.geometry_storage.push_back(create_gpu_obj3d(tracker));
    }
    gpu_geometry.geometry_view = axstd::span(gpu_geometry.geometry_storage.data(), gpu_geometry.geometry_storage.size());
  }

  template<>
  void DispatchedGeometryReferenceStorage<DevicePolicy>::mapResources() {
    for (auto &tracker : gpu_geometry.buffers_trackers) {
      map_gpu_resources(tracker);
    }
  }

  template<>
  void DispatchedGeometryReferenceStorage<DevicePolicy>::release() {
    for (auto &tracker : gpu_geometry.buffers_trackers) {
      unmap_gpu_resources(tracker);
    }
  }

  template<>
  void DispatchedGeometryReferenceStorage<DevicePolicy>::clear() {
    cpu_geometry.geometry_storage.clear();
    gpu_geometry.geometry_storage.clear();
    gpu_geometry.buffers_trackers.clear();
  }

  template<>
  mesh_vertex_attrib_views_t DispatchedGeometryReferenceStorage<DevicePolicy>::getGeometryViews() const {
    mesh_vertex_attrib_views_t views;
    views.host_geometry_view = cpu_geometry.geometry_storage;
    views.device_geometry_view = gpu_geometry.geometry_storage;
    return views;
  }

  /***************************************************************************************************************/

  template<>
  void DispatchedGeometryReferenceStorage<HostPolicy>::allocate(std::size_t num_meshes) {
    cpu_geometry.geometry_storage.reserve(num_meshes);
    container_capacity = num_meshes;
  }

  template<>
  void DispatchedGeometryReferenceStorage<HostPolicy>::mapResources() {}

  template<>
  void DispatchedGeometryReferenceStorage<HostPolicy>::release() {}

  template<>
  std::size_t DispatchedGeometryReferenceStorage<>::addGeometry(const Object3D &geometry) {
    cpu_geometry.geometry_storage.push_back(geometry);
    return cpu_geometry.geometry_storage.size() - 1;
  }

  template<>
  void DispatchedGeometryReferenceStorage<HostPolicy>::clear() {
    cpu_geometry.geometry_storage.clear();
  }

  template<>
  mesh_vertex_attrib_views_t DispatchedGeometryReferenceStorage<HostPolicy>::getGeometryViews() const {
    mesh_vertex_attrib_views_t views;
    views.host_geometry_view = cpu_geometry.geometry_storage;
    views.device_geometry_view = {};
    return views;
  }

  template<>
  const IdxMeshesView &DispatchedGeometryReferenceStorage<HostPolicy>::getGPUBuffersView() const {
    return gpu_geometry.geometry_view;
  }

  template<>
  void DispatchedGeometryReferenceStorage<HostPolicy>::mapBuffers() {
    cpu_geometry.geometry_view = axstd::span(cpu_geometry.geometry_storage.data(), cpu_geometry.geometry_storage.size());
  }

  template<>
  const IdxMeshesView &DispatchedGeometryReferenceStorage<>::getCPUBuffersView() const {
    return cpu_geometry.geometry_view;
  }

}  // namespace nova::shape::triangle

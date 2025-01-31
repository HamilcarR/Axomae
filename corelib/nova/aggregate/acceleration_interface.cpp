#include "acceleration_interface.h"
#include "embree4/rtcore_common.h"
#include "embree4/rtcore_device.h"
#include "embree4/rtcore_geometry.h"
#include "embree4/rtcore_scene.h"
#include "glm/gtc/type_ptr.hpp"
#include "internal/macro/project_macros.h"
#include "primitive/PrimitiveInterface.h"
#include "shape/shape_datastructures.h"
#include <cstring>
#include <embree4/rtcore.h>
#include <internal/debug/Logger.h>
namespace nova::aggregate {

  struct error_data_t {
    std::size_t mesh_index;
    std::size_t transform_offset;
  };
  void embree_error_callback(void *userPtr, RTCError error, const char *str) {
    std::string error_string = std::string("Encountered Embree error: ") + std::to_string(error) + "  " + str;
    error_data_t *error_status = static_cast<error_data_t *>(userPtr);
    std::string primitive_err_string = std::string("Mesh index: ") + std::to_string(error_status->mesh_index) +
                                       std::string(" and transform offset : ") + std::to_string(error_status->transform_offset);
    std::string final_err = error_string + primitive_err_string;
    if (error == RTC_ERROR_UNKNOWN) {
      LOG("Fatal error :" + final_err, LogLevel::CRITICAL);
      exit(1);
    }
    LOG("Embree error : " + final_err, LogLevel::ERROR);
  }

  class GenericAccelerator::Impl {
    RTCDevice device;
    RTCScene root_scene;
    error_data_t error_status{};
    primitive_aggregate_data_s
        primitive_list;  // contains meshes geometry , transformations , and the list of primitives abstractions to return when ray hits
   public:
    Impl() {
      device = rtcNewDevice(nullptr);
      root_scene = createScene();
      rtcSetDeviceErrorFunction(device, embree_error_callback, &error_status);
    }

    RTCGeometry createTriangleGeometry() { return rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE); }

    RTCGeometry createInstanceGeometry() { return rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE); }

    RTCScene createScene() { return rtcNewScene(device); }

    void commit(RTCGeometry &mesh) { rtcCommitGeometry(mesh); }

    void commit(RTCScene &scene) { rtcCommitScene(scene); }

    unsigned attach(RTCScene &scene, RTCGeometry &mesh) { return rtcAttachGeometry(scene, mesh); }

    void release(RTCGeometry &mesh) { rtcReleaseGeometry(mesh); }

    void initializeGeometry(RTCGeometry &geom, const Object3D &mesh) {
      float *vertices = static_cast<float *>(
          rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT, sizeof(float), mesh.vertices.size()));
      std::memcpy(vertices, mesh.vertices.data(), mesh.vertices.size() * sizeof(float));

      unsigned *indices = static_cast<unsigned *>(
          rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT4, sizeof(unsigned), mesh.indices.size()));
      std::memcpy(indices, mesh.indices.data(), mesh.indices.size() * sizeof(unsigned));

      float *normals = static_cast<float *>(
          rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_NORMAL, 0, RTC_FORMAT_FLOAT, sizeof(float), mesh.normals.size()));
      std::memcpy(normals, mesh.normals.data(), mesh.normals.size() * sizeof(float));

      float *tangents = static_cast<float *>(
          rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_TANGENT, 0, RTC_FORMAT_FLOAT, sizeof(float), mesh.tangents.size()));
      std::memcpy(tangents, mesh.tangents.data(), mesh.tangents.size() * sizeof(float));
    }

    void setInstancedScene(RTCGeometry &instance, RTCScene &scene, unsigned timestep = 1) {
      rtcSetGeometryInstancedScene(instance, scene);
      rtcSetGeometryTimeStepCount(instance, timestep);
    }

    /* Data format is column major.*/
    void setGeometryTransform4x4(RTCGeometry &geom, const float *transform) {
      rtcSetGeometryTransform(geom, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform);
    }

    void cleanup() { rtcReleaseScene(root_scene); }

    void setupTransformedMeshes(primitive_aggregate_data_s primitive_aggregate) {
      AX_ASSERT_NOTNULL(primitive_aggregate.mesh_geometry);
      primitive_list = primitive_aggregate;

      const shape::mesh_shared_views_t &mesh_geometry = *primitive_aggregate.mesh_geometry;
      for (int mesh_index = 0; mesh_index < mesh_geometry.geometry.host_geometry_view.size(); mesh_index++) {
        const Object3D &mesh = mesh_geometry.geometry.host_geometry_view[mesh_index];
        std::size_t transform_offset = mesh_geometry.transforms.mesh_offsets_to_matrix[mesh_index];
        shape::transform::transform4x4_t transform{};
        int err = shape::transform::reconstruct_transform4x4(transform, transform_offset, mesh_geometry.transforms);
        AX_ASSERT_EQ(err, 0);
        error_status.transform_offset = transform_offset;
        error_status.mesh_index = mesh_index;
        setupTransformedMesh(mesh, transform);
      }
      commit(root_scene);
    }

    void setupTransformedMesh(const Object3D &mesh_geometry, const shape::transform::transform4x4_t &transform) {
      RTCGeometry geometry = createTriangleGeometry();
      RTCScene subscene = createScene();
      initializeGeometry(geometry, mesh_geometry);
      commit(geometry);
      attach(subscene, geometry);
      release(geometry);
      commit(subscene);
      /* Transformations. */
      RTCGeometry instance = createInstanceGeometry();
      setInstancedScene(instance, subscene);
      setGeometryTransform4x4(instance, glm::value_ptr(transform.m));
      commit(instance);
      attach(root_scene, instance);
      release(instance);
    }

    static RTCRayHit translate(const Ray &ray) {
      RTCRayHit hit{};

      hit.ray.org_x = ray.origin.x;
      hit.ray.org_y = ray.origin.y;
      hit.ray.org_z = ray.origin.z;

      hit.ray.dir_x = ray.direction.x;
      hit.ray.dir_y = ray.direction.y;
      hit.ray.dir_z = ray.direction.z;

      hit.ray.tnear = ray.tnear;
      hit.ray.tfar = ray.tfar;
      hit.ray.mask = -1;
      hit.ray.flags = 0;
      hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
      hit.hit.primID = RTC_INVALID_GEOMETRY_ID;
      hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

      return hit;
    }
    bool intersect(const Ray &ray, bvh_hit_data &hit_return_data) {

      RTCRayHit result = translate(ray);
      RTCRayQueryContext context{};
      rtcInitRayQueryContext(&context);
      RTCIntersectArguments intersect_args{};
      rtcInitIntersectArguments(&intersect_args);
      intersect_args.context = &context;

      rtcIntersect1(root_scene, &result, &intersect_args);

      if (result.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        hit_return_data.is_hit = true;
        hit_return_data.prim_min_t = result.ray.tfar;
        hit_return_data.prim_max_t = ray.tfar;
        hit_return_data.hit_d.t = result.ray.tfar;
        hit_return_data.hit_d.u = result.hit.u;
        hit_return_data.hit_d.v = result.hit.v;
        hit_return_data.hit_d.position = ray.pointAt(result.ray.tnear);
        const primitives_view_tn &temp_prim_view = *primitive_list.primitive_list_view;
        hit_return_data.last_primit = &temp_prim_view[result.hit.instID[0]];
        hit_return_data.hit_d.normal = glm::normalize(glm::vec3(result.hit.Ng_x, result.hit.Ng_y, result.hit.Ng_z));  // Object space
        return true;
      } else {
        hit_return_data.is_hit = false;
        return false;
      }
    }
  };

  /****************************************************************************************************************************************************************************/

  GenericAccelerator::GenericAccelerator() : pimpl(std::make_unique<Impl>()) {}
  GenericAccelerator::~GenericAccelerator() = default;
  GenericAccelerator::GenericAccelerator(GenericAccelerator &&) noexcept = default;
  GenericAccelerator &GenericAccelerator::operator=(GenericAccelerator &&) noexcept = default;

  void GenericAccelerator::build(primitive_aggregate_data_s meshes) { pimpl->setupTransformedMeshes(meshes); }

  bool GenericAccelerator::hit(const Ray &ray, bvh_hit_data &hit_data) const { return pimpl->intersect(ray, hit_data); }

  void GenericAccelerator::cleanup() { pimpl->cleanup(); }

}  // namespace nova::aggregate

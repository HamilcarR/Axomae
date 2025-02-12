#include "acceleration_interface.h"
#include "primitive/PrimitiveInterface.h"
#include "ray/Hitable.h"
#include "shape/shape_datastructures.h"
#include <cstring>
#include <embree4/rtcore.h>
#include <embree4/rtcore_common.h>
#include <embree4/rtcore_device.h>
#include <embree4/rtcore_geometry.h>
#include <embree4/rtcore_ray.h>
#include <embree4/rtcore_scene.h>
#include <internal/common/math/utils_3D.h>
#include <internal/debug/Logger.h>
#include <internal/geometry/Object3D.h>
#include <internal/macro/project_macros.h>
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
    std::string final_err = error_string + "\n" + primitive_err_string;
    if (error == RTC_ERROR_UNKNOWN) {
      LOG("Fatal error :" + final_err, LogLevel::CRITICAL);
      exit(1);
    }
    LOG("Embree error : " + final_err, LogLevel::ERROR);
  }

  template<>
  class GenericAccelerator<EmbreeBuild>::Impl {
    RTCDevice device;
    RTCScene root_scene{};
    RTCScene instanced_scene{};
    error_data_t error_status{};
    /*contains meshes geometry , transformations , and the list of primitives abstractions to return when ray hits.*/
    primitive_aggregate_data_s scene_geometry_data{};
    /* Computes the global offset of primitives for each mesh.
     * If mesh i contains X primitives , and mesh i - 1 contains Y primitives , then
     * primitive_global_offset[i] = X + Y. Knowing this , we can access the global offset of the hit
     * primitive doing : primitive_global_offset[geomID - 1] + primID
     */
    std::vector<std::size_t> primitive_global_offset;
    bool has_valid_tangents{false};
    bool has_valid_bitangents{false};
    bool has_valid_normals{false};
    bool has_valid_uvs{false};

   public:
    Impl() {
      device = rtcNewDevice(nullptr);
      rtcSetDeviceErrorFunction(device, embree_error_callback, &error_status);
    }

    ~Impl() {
      cleanup();
      if (device)
        rtcReleaseDevice(device);
      device = nullptr;
    }

    Impl(const Impl &) = default;
    Impl(Impl &&) noexcept = default;
    Impl &operator=(const Impl &) = default;
    Impl &operator=(Impl &&) noexcept = default;

    void cleanup() {
      if (instanced_scene)
        rtcReleaseScene(instanced_scene);
      if (root_scene)
        rtcReleaseScene(root_scene);
      instanced_scene = nullptr;
      root_scene = nullptr;
    }

    RTCGeometry createTriangleGeometry() { return rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE); }
    RTCGeometry createInstanceGeometry() { return rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE); }
    RTCScene createScene() { return rtcNewScene(device); }
    void commit(RTCGeometry &mesh) { rtcCommitGeometry(mesh); }
    void commit(RTCScene &scene) { rtcCommitScene(scene); }
    unsigned attach(RTCScene &scene, RTCGeometry &mesh) { return rtcAttachGeometry(scene, mesh); }
    void release(RTCGeometry &mesh) { rtcReleaseGeometry(mesh); }
    void allocateAttribSlots(RTCGeometry &geom, int max_slots) { rtcSetGeometryVertexAttributeCount(geom, max_slots); }

    void initializeGeometry(RTCGeometry &geom, const Object3D &mesh) {
      allocateAttribSlots(geom, 4);
      rtcSetSharedGeometryBuffer(
          geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, mesh.vertices.data(), 0, 3 * sizeof(float), mesh.vertices.size() / 3);
      rtcSetSharedGeometryBuffer(
          geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, mesh.indices.data(), 0, 3 * sizeof(unsigned), mesh.indices.size() / 3);
      rtcSetSharedGeometryBuffer(
          geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT3, mesh.normals.data(), 0, 3 * sizeof(float), mesh.normals.size() / 3);
      rtcSetSharedGeometryBuffer(
          geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 1, RTC_FORMAT_FLOAT3, mesh.tangents.data(), 0, 3 * sizeof(float), mesh.tangents.size() / 3);
      rtcSetSharedGeometryBuffer(
          geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 2, RTC_FORMAT_FLOAT3, mesh.bitangents.data(), 0, 3 * sizeof(float), mesh.bitangents.size() / 3);
      rtcSetSharedGeometryBuffer(
          geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 3, RTC_FORMAT_FLOAT2, mesh.uv.data(), 0, 2 * sizeof(float), mesh.uv.size() / 2);
    }

    void setInstancedScene(RTCGeometry &instance, RTCScene &scene, unsigned timestep = 1) {
      rtcSetGeometryInstancedScene(instance, scene);
      rtcSetGeometryTimeStepCount(instance, timestep);
    }

    /* Data format is column major.*/
    void setGeometryTransform4x4(const RTCGeometry &geom, const float *transform) {
      rtcSetGeometryTransform(geom, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, transform);
    }

    shape::transform::transform4x4_t getTransform(std::size_t mesh_index) {
      const shape::mesh_shared_views_t &mesh_geometry = scene_geometry_data.mesh_geometry;
      std::size_t transform_offset = mesh_geometry.transforms.mesh_offsets_to_matrix[mesh_index];
      shape::transform::transform4x4_t transform{};
      int err = shape::transform::reconstruct_transform4x4(transform, transform_offset, mesh_geometry.transforms);
      AX_ASSERT_EQ(err, 0);
      return transform;
    }

    void setupTransformedMeshes(primitive_aggregate_data_s primitive_aggregate) {
      root_scene = createScene();
      instanced_scene = createScene();
      scene_geometry_data = primitive_aggregate;
      const shape::mesh_shared_views_t &mesh_geometry = scene_geometry_data.mesh_geometry;
      primitive_global_offset.resize(mesh_geometry.geometry.host_geometry_view.size());
      for (int mesh_index = 0; mesh_index < mesh_geometry.geometry.host_geometry_view.size(); mesh_index++) {
        const Object3D &mesh = mesh_geometry.geometry.host_geometry_view[mesh_index];
        std::size_t transform_offset = mesh_geometry.transforms.mesh_offsets_to_matrix[mesh_index];
        shape::transform::transform4x4_t transform{};
        int err = shape::transform::reconstruct_transform4x4(transform, transform_offset, mesh_geometry.transforms);
        AX_ASSERT_EQ(err, 0);
        error_status.transform_offset = transform_offset;
        error_status.mesh_index = mesh_index;
        unsigned geomID = setupTransformedMesh(mesh, transform);
        std::size_t triangles_number = mesh.indices.size() / 3 - 1;
        if (geomID == 0)
          primitive_global_offset[geomID] = triangles_number;  // Stores the number of primitives for each geomID.
        else
          primitive_global_offset[geomID] = primitive_global_offset[geomID - 1] + triangles_number;
      }
      commit(root_scene);
    }

    unsigned setupTransformedMesh(const Object3D &mesh_geometry, const shape::transform::transform4x4_t &transform) {
      RTCGeometry geometry = createTriangleGeometry();

      unsigned geomID = 0;
      initializeGeometry(geometry, mesh_geometry);
      commit(geometry);
      attach(instanced_scene, geometry);
      release(geometry);
      commit(instanced_scene);
      /* Transformations. */
      RTCGeometry instance = createInstanceGeometry();
      setInstancedScene(instance, instanced_scene);
      setGeometryTransform4x4(instance, glm::value_ptr(transform.m));
      commit(instance);
      geomID = attach(root_scene, instance);
      release(instance);
      return geomID;
    }

    static RTCRayHit apiRay(const Ray &ray) {
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

    const nova::primitive::NovaPrimitiveInterface *getHitPrimitiveFromOffset(const RTCRayHit &result) {
      const primitives_view_tn &temp_prim_view = scene_geometry_data.primitive_list_view;
      std::size_t primitive_offset = 0;
      if (result.hit.geomID == 0)
        primitive_offset = result.hit.primID;
      else
        primitive_offset = primitive_global_offset[result.hit.geomID - 1] + result.hit.primID;
      std::size_t global_prim_id = primitive_offset;
      return &temp_prim_view[global_prim_id];
    }

    inline shape::transform::transform4x4_t getMeshTransform(unsigned geomID) {
      shape::transform::transform4x4_t transform{};
      std::size_t transform_offset = scene_geometry_data.mesh_geometry.transforms.mesh_offsets_to_matrix[geomID];
      shape::transform::reconstruct_transform4x4(transform, transform_offset, scene_geometry_data.mesh_geometry.transforms);
      return transform;
    }

    bool intersect(const Ray &ray, bvh_hit_data &hit_return_data) {
      RTCRayHit result = apiRay(ray);
      RTCRayQueryContext context{};
      rtcInitRayQueryContext(&context);
      RTCIntersectArguments intersect_args{};
      rtcInitIntersectArguments(&intersect_args);
      intersect_args.context = &context;
      rtcIntersect1(root_scene, &result, &intersect_args);

      if (result.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        hit_return_data.is_hit = true;
        /* hit distance t is registered in tfar for embree.*/
        hit_return_data.prim_min_t = result.ray.tfar;
        hit_return_data.prim_max_t = ray.tfar;
        hit_return_data.hit_d.t = result.ray.tfar;
        hit_return_data.hit_d.position = ray.pointAt(result.ray.tfar);
        hit_return_data.last_primit = getHitPrimitiveFromOffset(result);
        float normals[3]{}, tangents[3]{}, bitangents[3]{}, uv[2]{};
        RTCGeometry this_geometry = rtcGetGeometry(instanced_scene, result.hit.geomID);
        rtcInterpolate0(this_geometry, result.hit.primID, result.hit.u, result.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, normals, 3);
        rtcInterpolate0(this_geometry, result.hit.primID, result.hit.u, result.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 1, tangents, 3);
        rtcInterpolate0(this_geometry, result.hit.primID, result.hit.u, result.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 2, bitangents, 3);
        rtcInterpolate0(this_geometry, result.hit.primID, result.hit.u, result.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 3, uv, 2);
        auto transform = getMeshTransform(result.hit.geomID);
        hit_return_data.hit_d.u = std::fabs(uv[1]);
        hit_return_data.hit_d.v = std::fabs(uv[0]);
        hit_return_data.hit_d.normal = glm::normalize(transform.n * glm::vec3(normals[0], normals[1], normals[2]));
        hit_return_data.hit_d.tangent = glm::normalize(transform.n * glm::vec3(tangents[0], tangents[1], tangents[2]));
        hit_return_data.hit_d.bitangent = glm::normalize(transform.n * glm::vec3(bitangents[0], bitangents[1], bitangents[2]));

        return true;
      } else {
        hit_return_data.is_hit = false;
        return false;
      }
    }
  };

  /****************************************************************************************************************************************************************************/
  template<>
  GenericAccelerator<EmbreeBuild>::GenericAccelerator() : pimpl(std::make_unique<Impl>()) {}
  template<>
  GenericAccelerator<EmbreeBuild>::~GenericAccelerator() {}
  template<>
  GenericAccelerator<EmbreeBuild>::GenericAccelerator(GenericAccelerator &&) noexcept = default;
  template<>
  GenericAccelerator<EmbreeBuild> &GenericAccelerator<EmbreeBuild>::operator=(GenericAccelerator &&) noexcept = default;
  template<>
  void GenericAccelerator<EmbreeBuild>::build(primitive_aggregate_data_s meshes) {
    pimpl->setupTransformedMeshes(meshes);
  }
  template<>
  bool GenericAccelerator<EmbreeBuild>::hit(const Ray &ray, bvh_hit_data &hit_data) const {
    return pimpl->intersect(ray, hit_data);
  }
  template<>
  void GenericAccelerator<EmbreeBuild>::cleanup() {
    pimpl->cleanup();
  }

}  // namespace nova::aggregate

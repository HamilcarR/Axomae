#include "../../acceleration_interface.h"
#include "../../aggregate_datastructures.h"
#include "aggregate/AccelerationInternalsInterface.h"
#include "glm/gtc/type_ptr.hpp"
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
#include <internal/common/math/math_includes.h>
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
    std::string primitive_err_string = "";
    if (error_status) {
      primitive_err_string = std::string("Mesh index: ") + std::to_string(error_status->mesh_index) + std::string(" and transform offset : ") +
                             std::to_string(error_status->transform_offset);
    }
    std::string final_err = error_string + "\n" + primitive_err_string;
    if (error == RTC_ERROR_UNKNOWN) {
      LOG("Fatal error :" + final_err, LogLevel::CRITICAL);
      exit(1);
    }
    LOG("Embree error : " + final_err, LogLevel::ERROR);
  }

  inline const nova::primitive::NovaPrimitiveInterface *getPrimitiveFromGlobalOffset(std::size_t geometry_id,
                                                                                     std::size_t primitive_id,
                                                                                     const std::vector<std::size_t> &primitive_global_offset,
                                                                                     const primitive_aggregate_data_s &scene) {
    const primitive::primitives_view_tn &temp_prim_view = scene.primitive_list_view;
    std::size_t primitive_offset = 0;
    if (geometry_id == 0)
      primitive_offset = primitive_id;
    else
      primitive_offset = primitive_global_offset[geometry_id - 1] + primitive_id;
    return &temp_prim_view[primitive_offset];
  }
  /* Computes the global offset of primitives for each mesh.
   * If mesh i contains X primitives , and mesh i - 1 contains Y primitives , then
   * primitive_global_offset[i] = X + Y. Knowing this , we can access the global offset of the hit
   * primitive doing : primitive_global_offset[geomID - 1] + primID
   */
  inline void setupGlobalOffsetArray(const primitive_aggregate_data_s &scene_geometry_data, std::vector<std::size_t> &primitive_global_offset) {
    const auto &mesh_geometry = scene_geometry_data.mesh_geometry.getTriangleGeometryViews();
    primitive_global_offset.resize(mesh_geometry.size());
    for (int geometry_index = 0; geometry_index < mesh_geometry.size(); geometry_index++) {
      const Object3D &mesh = mesh_geometry[geometry_index];
      AX_ASSERT_EQ(mesh.face_stride, 3);  // Check that it's a triangle. Other primitives support need to be done later.
      std::size_t primitive_count = mesh.indices.size() / mesh.face_stride;
      if (geometry_index == 0)
        primitive_global_offset[geometry_index] = primitive_count;  // Stores the number of primitives for each geomID.
      else
        primitive_global_offset[geometry_index] = primitive_global_offset[geometry_index - 1] + primitive_count;
    }
  }

  template<class T>
  void validate3(const T *value) {
    AX_ASSERT_NOTNULL(value);
    AX_ASSERT(!ISNAN(value[0]) && !ISNAN(value[1]) && !ISNAN(value[2]), "");
  }

  template<class T>
  void validate2(const T *value) {
    AX_ASSERT_NOTNULL(value);
    AX_ASSERT(!ISNAN(value[0]) && !ISNAN(value[1]), "");
  }

  template<class T>
  void validate1(T value) {
    AX_ASSERT(!ISNAN(value), "");
  }

  template<class T>
  void validate9(const T *value) {
    AX_ASSERT_NOTNULL(value);
    for (unsigned i = 0; i < 9; i++)
      AX_ASSERT(!ISNAN(value[i]), "");
  }

  template<class T>
  void assert_valid(T value[3]) {
    AX_ASSERT_NOTNULL(value);
    AX_ASSERT(value[0] != 0 || value[1] != 0 || value[2] != 0, "");
    AX_ASSERT_NOTNAN(value[0]);
    AX_ASSERT_NOTNAN(value[1]);
    AX_ASSERT_NOTNAN(value[2]);
  }

#define acceleration_internal_interface \
 protected \
  AccelerationInternalsInterface<GenericHostAccelerator<EmbreeBuild>::Impl>

  template<>
  class GenericHostAccelerator<EmbreeBuild>::Impl : acceleration_internal_interface {

    enum FLAG_VERTEX_ATTRIB { NORMAL = 0, TANGENT = 1, BITANGENT = 2, UV = 3 };

    struct user_geometry_structure_s {
      std::size_t transform_offset;
      std::size_t mesh_index;
    };

    RTCDevice device;
    RTCScene root_scene{};
    std::vector<RTCScene> mesh_scenes{};  // Keep references to subscenes for lookup in interpolation method.
    /*contains meshes geometry , transformations , and the list of primitives abstractions to return when ray hits.*/
    primitive_aggregate_data_s scene_geometry_data{};
    std::vector<std::size_t> primitive_global_offset;
    std::vector<user_geometry_structure_s> user_geometry;

   public:
    Impl() {
      device = rtcNewDevice(nullptr);
      rtcSetDeviceErrorFunction(device, embree_error_callback, nullptr);
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

    void build(primitive_aggregate_data_s primitive_aggregate) {
      root_scene = rtcNewScene(device);
      scene_geometry_data = primitive_aggregate;
      const shape::MeshBundleViews &mesh_geometry = scene_geometry_data.mesh_geometry;
      const std::size_t mesh_number = mesh_geometry.getTriangleGeometryViews().size();

      initializeApplicationStructures(mesh_number);

      for (int mesh_index = 0; mesh_index < mesh_number; mesh_index++) {
        const Object3D &current_mesh = mesh_geometry.getTriangleMesh(mesh_index);
        std::size_t transform_offset = mesh_geometry.getTransformOffset(mesh_index);

        transform4x4_t transform = mesh_geometry.reconstructTransform4x4(mesh_index);

        RTCGeometry inst_geometry = addInstancedTriangleMesh(current_mesh, transform, mesh_index);

        user_geometry_structure_s user_data{};
        user_data.mesh_index = mesh_index;
        user_data.transform_offset = transform_offset;
        user_geometry[mesh_index] = user_data;
        rtcSetGeometryUserData(inst_geometry, &user_geometry[mesh_index]);

        rtcReleaseGeometry(inst_geometry);
      }
    }

    bool intersect(const Ray &ray, bvh_hit_data &hit_return_data) const {
      RTCRayHit result = apiRay(ray);
      RTCRayQueryContext context{};
      rtcInitRayQueryContext(&context);
      RTCIntersectArguments intersect_args{};
      rtcInitIntersectArguments(&intersect_args);
      intersect_args.context = &context;
      rtcIntersect1(root_scene, &result, &intersect_args);

      if (result.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
        fillHitStructure(ray, result, hit_return_data);
#ifndef NDEBUG
        validate(hit_return_data);
#endif
        return true;
      } else {
        hit_return_data.is_hit = false;
        return false;
      }
    }

    void cleanup() {
      primitive_global_offset.clear();
      user_geometry.clear();

      for (auto &e : mesh_scenes)
        if (e)
          rtcReleaseScene(e);
      mesh_scenes.clear();

      if (root_scene)
        rtcReleaseScene(root_scene);
      root_scene = nullptr;
    }

   private:
    void initializeApplicationStructures(std::size_t mesh_number) {
      user_geometry.resize(mesh_number);
      mesh_scenes.resize(mesh_number);
      setupGlobalOffsetArray(scene_geometry_data, primitive_global_offset);
    }

    void initializeGeometry(RTCGeometry &geom, const Object3D &mesh) {
      rtcSetGeometryVertexAttributeCount(geom, 4);
      rtcSetSharedGeometryBuffer(
          geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, mesh.vertices.data(), 0, 3 * sizeof(float), mesh.vertices.size() / 3);
      rtcSetSharedGeometryBuffer(
          geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, mesh.indices.data(), 0, 3 * sizeof(unsigned), mesh.indices.size() / 3);
      rtcSetSharedGeometryBuffer(
          geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, NORMAL, RTC_FORMAT_FLOAT3, mesh.normals.data(), 0, 3 * sizeof(float), mesh.normals.size() / 3);
      rtcSetSharedGeometryBuffer(
          geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, TANGENT, RTC_FORMAT_FLOAT3, mesh.tangents.data(), 0, 3 * sizeof(float), mesh.tangents.size() / 3);
      rtcSetSharedGeometryBuffer(geom,
                                 RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE,
                                 BITANGENT,
                                 RTC_FORMAT_FLOAT3,
                                 mesh.bitangents.data(),
                                 0,
                                 3 * sizeof(float),
                                 mesh.bitangents.size() / 3);
      rtcSetSharedGeometryBuffer(
          geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, UV, RTC_FORMAT_FLOAT2, mesh.uv.data(), 0, 2 * sizeof(float), mesh.uv.size() / 2);
    }

    RTCGeometry addInstancedTriangleMesh(const Object3D &mesh_geometry, const transform4x4_t &transform, std::size_t mesh_index) {
      RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
      initializeGeometry(geometry, mesh_geometry);
      RTCScene sub_scene = rtcNewScene(device);
      rtcCommitGeometry(geometry);
      rtcAttachGeometryByID(sub_scene, geometry, mesh_index);
      rtcReleaseGeometry(geometry);
      rtcCommitScene(sub_scene);
      /* Transformations. */
      RTCGeometry instance_geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_INSTANCE);
      rtcSetGeometryInstancedScene(instance_geometry, sub_scene);
      rtcSetGeometryTimeStepCount(instance_geometry, 1);
      rtcSetGeometryTransform(instance_geometry, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, glm::value_ptr(transform.m));
      rtcCommitGeometry(instance_geometry);
      rtcAttachGeometry(root_scene, instance_geometry);
      rtcCommitScene(root_scene);
      mesh_scenes[mesh_index] = sub_scene;
      return instance_geometry;
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

    void validate(const bvh_hit_data &hit_return_data) const {
      hit_geometry_s geometry = hit_return_data.hit_d.geometry;
      hit_shading_s shading = hit_return_data.hit_d.shading;
      validate3(glm::value_ptr(geometry.position));
      validate1(geometry.t);
      validate1(geometry.u);
      validate1(geometry.v);
    }

    hit_geometry_s computeDerivativesTriangle(
        const Ray &ray, const RTCRayHit &hit_data, const shape::face_data_s &face, const transform4x4_t &transform, float u, float v) const {
      AX_ASSERT(face.type == shape::FACE::TRIANGLE, "Invalid shape type.");

      hit_geometry_s dpd{};
      glm::mat3 l_transform = glm::mat3(transform.m);
      geometry::face_data_tri triface = face.data.triangle_face;
      float duv02x = triface.uv0[0] - triface.uv2[0], duv02y = triface.uv0[1] - triface.uv2[1];
      float duv12x = triface.uv1[0] - triface.uv2[0], duv12y = triface.uv1[1] - triface.uv2[1];

      glm::vec2 duv02{duv02x, duv02y};
      glm::vec2 duv12{duv12x, duv12y};

      float dp02x = triface.v0[0] - triface.v2[0], dp02y = triface.v0[1] - triface.v2[1], dp02z = triface.v0[2] - triface.v2[2];
      float dp12x = triface.v1[0] - triface.v2[0], dp12y = triface.v1[1] - triface.v2[1], dp12z = triface.v1[2] - triface.v2[2];

      glm::vec3 dp02 = l_transform * glm::vec3(dp02x, dp02y, dp02z);
      glm::vec3 dp12 = l_transform * glm::vec3(dp12x, dp12y, dp12z);

      // computes det(duv02 , duv12)
      float determinant = duv02x * duv12y - duv02y * duv12x;

      bool isDegenUV = fabsf(determinant) < 1e-9f;

      float invdet = 1.f / determinant;
      glm::vec3 dpdu = (duv12y * dp02 - duv02y * dp12) * invdet;
      glm::vec3 dpdv = (duv02x * dp12 - duv12x * dp02) * invdet;

      glm::vec3 n;

      if (isDegenUV || glm::length(glm::cross(dpdu, dpdv)) == 0) {
        n = glm::normalize(glm::cross(dp12, dp02));
        math::make_onb(n, dpdu, dpdv);
        dpd.degenerate = true;
      } else {
        dpd.degenerate = false;
        dpd.e1 = dp02;
        dpd.e2 = dp12;
        n = glm::cross(dpd.e1, dpd.e2);  // Computes true geometric normal right handed.
      }
      AX_ASSERT_NOTNAN(n);

      dpd.ng = glm::normalize(n);
      dpd.position = ray.pointAt(hit_data.ray.tfar);
      dpd.dpdu = dpdu;
      dpd.dpdv = dpdv;
      AX_ASSERT(dpd.dpdu.x != 0 || dpd.dpdu.y != 0 || dpd.dpdu.z != 0, "");
      AX_ASSERT(dpd.dpdv.x != 0 || dpd.dpdv.y != 0 || dpd.dpdv.z != 0, "");
      dpd.u = u;
      dpd.v = v;
      dpd.t = hit_data.ray.tfar;
      dpd.wo_dot_n = glm::dot(-glm::normalize(ray.direction), dpd.ng);
      return dpd;
    }

    hit_geometry_s fillGeometryStructure(const Ray &ray,
                                         const RTCGeometry &this_geometry,
                                         const RTCRayHit &result,
                                         const primitive::NovaPrimitiveInterface *last_primit,
                                         const transform4x4_t &transform) const {
      shape::face_data_s face = last_primit->getFace(scene_geometry_data.mesh_geometry);
      float uv[2]{};
      rtcInterpolate0(this_geometry, result.hit.primID, result.hit.u, result.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, UV, uv, 2);
      hit_geometry_s geometry = computeDerivativesTriangle(ray, result, face, transform, uv[0], uv[1]);
      return geometry;
    }

    struct frame_components_s {
      glm::vec3 normal;
      glm::vec3 tangent;
      glm::vec3 bitangent;
    };

    frame_components_s fillShadingStructure(const RTCGeometry &this_geometry, const RTCRayHit &result, const transform4x4_t &transform) const {
      frame_components_s shading{};
      float normals[3]{}, tangents[3]{}, bitangents[3]{};

      rtcInterpolate0(this_geometry, result.hit.primID, result.hit.u, result.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, NORMAL, normals, 3);
      rtcInterpolate0(this_geometry, result.hit.primID, result.hit.u, result.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, TANGENT, tangents, 3);
      rtcInterpolate0(this_geometry, result.hit.primID, result.hit.u, result.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, BITANGENT, bitangents, 3);

      glm::vec3 transformed_normal = glm::normalize(transform.n * glm::vec3(normals[0], normals[1], normals[2]));
      glm::vec3 transformed_tangent = glm::normalize(glm::mat3(transform.m) * glm::vec3(tangents[0], tangents[1], tangents[2]));
      glm::vec3 transformed_bitangent = glm::normalize(glm::mat3(transform.m) * glm::vec3(bitangents[0], bitangents[1], bitangents[2]));
#ifndef NDEBUG
      assert_valid(normals);
      assert_valid(tangents);
      assert_valid(bitangents);
#endif

      shading.normal = transformed_normal;
      shading.bitangent = transformed_bitangent;
      shading.tangent = transformed_tangent;

      return shading;
    }

    hit_shading_s constructTbnFrame(const hit_geometry_s &geometry, const frame_components_s &shading_components) const {
      AX_ASSERT_NOTNAN(shading_components.normal);
      AX_ASSERT_NOTNAN(shading_components.tangent);
      hit_shading_s shading_frame;
      shading_frame.frame = IntersectFrame(shading_components.tangent, shading_components.bitangent, shading_components.normal, true);
      return shading_frame;
    }

    void fillHitStructure(const Ray &ray, const RTCRayHit &result, bvh_hit_data &hit_return_data) const {

      hit_return_data.is_hit = true;
      /* hit distance t is registered in tfar for embree.*/
      hit_return_data.prim_min_t = result.ray.tfar;
      hit_return_data.prim_max_t = ray.tfar;

      const primitive::NovaPrimitiveInterface *last_primitive = getPrimitiveFromGlobalOffset(
          result.hit.geomID, result.hit.primID, primitive_global_offset, scene_geometry_data);
      hit_return_data.last_primit = last_primitive;

      auto *user_g = static_cast<const user_geometry_structure_s *>(rtcGetGeometryUserDataFromScene(root_scene, result.hit.geomID));
      transform4x4_t transform = scene_geometry_data.mesh_geometry.reconstructTransform4x4(user_g->mesh_index);

      RTCGeometry this_geometry = rtcGetGeometry(mesh_scenes[result.hit.geomID], result.hit.instID[0]);
      hit_geometry_s geometry = fillGeometryStructure(ray, this_geometry, result, last_primitive, transform);
      hit_return_data.hit_d.geometry = geometry;

      hit_shading_s shading_frame = constructTbnFrame(geometry, fillShadingStructure(this_geometry, result, transform));
      hit_return_data.hit_d.shading = shading_frame;
    }
  };

  /****************************************************************************************************************************************************************************/
  template<>
  GenericHostAccelerator<EmbreeBuild>::GenericHostAccelerator() : pimpl(std::make_unique<Impl>()) {}
  template<>
  GenericHostAccelerator<EmbreeBuild>::~GenericHostAccelerator() = default;
  template<>
  GenericHostAccelerator<EmbreeBuild>::GenericHostAccelerator(GenericHostAccelerator &&) noexcept = default;
  template<>
  GenericHostAccelerator<EmbreeBuild> &GenericHostAccelerator<EmbreeBuild>::operator=(GenericHostAccelerator &&) noexcept = default;
  template<>
  void GenericHostAccelerator<EmbreeBuild>::build(primitive_aggregate_data_s meshes) {
    pimpl->build(meshes);
  }
  template<>
  bool GenericHostAccelerator<EmbreeBuild>::hit(const Ray &ray, bvh_hit_data &hit_data) const {
    return pimpl->intersect(ray, hit_data);
  }
  template<>
  void GenericHostAccelerator<EmbreeBuild>::cleanup() {
    pimpl->cleanup();
  }

}  // namespace nova::aggregate

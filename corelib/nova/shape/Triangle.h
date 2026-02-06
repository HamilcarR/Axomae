#ifndef TRIANGLE_H
#define TRIANGLE_H
#include "MeshContext.h"
#include "ray/Hitable.h"
#include "ray/IntersectFrame.h"
#include "ray/Ray.h"
#include "shape_datastructures.h"
#include <internal/common/math/math_utils.h>
#include <internal/common/math/utils_3D.h>
#include <internal/device/gpgpu/device_macros.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/geometry/BoundingBox.h>
#include <internal/geometry/Object3D.h>
#include <internal/macro/project_macros.h>

namespace nova::shape {

  struct mesh_properties_t {
    const Object3D *geometry;
    std::size_t transform_offset;
  };

  namespace triangle {

    /* Returns all vertex attributes from a triangle in a mesh.
     * Attributes that don't exist are nullptr.
     */
    ax_device_callable_inlined geometry::face_data_tri get_face(const Object3D *geometry, std::size_t triangle_id) {
      geometry::face_data_tri tri_primitive{};
      const auto &indices = geometry->indices;
      unsigned idx[3] = {indices[triangle_id], indices[triangle_id + 1], indices[triangle_id + 2]};
      geometry->getTri(tri_primitive, idx);
      return tri_primitive;
    }

    ax_device_callable_inlined geometry::face_data_tri get_face(const Object3D *geometry, const transform4x4_t &transform, std::size_t triangle_id) {
      geometry::face_data_tri tri_primitive{};
      const auto &indices = geometry->indices;
      unsigned idx[3] = {indices[triangle_id], indices[triangle_id + 1], indices[triangle_id + 2]};
      geometry->getTri(tri_primitive, idx);
      tri_primitive.transform(transform.m, transform.n);
      return tri_primitive;
    }

  }  // namespace triangle

  class Triangle {
   private:
    /* Index of the triangle in the corresponding mesh_id's geometry.
     * Vertices are accessed with triangle_id + 0 , triangle_id + 1 ,triangle_id + 2 in the index array (MeshCtx). Must be divisible by 3.
     * ex : triangle n°3 is index[3 * 3] , index[3 * 4] , index[3 * 5].
     */
    uint32_t triangle_id{};
    uint32_t mesh_id{};

   public:
    CLASS_DCM(Triangle)
    ax_device_callable_inlined face_data_s getFace(const MeshCtx &geometry) const {
      const Object3D &obj3d = geometry.getTriMesh(mesh_id);
      geometry::face_data_tri face = triangle::get_face(&obj3d, triangle_id);
      face_data_s face_d{};
      face_d.type = TRIANGLE;
      face_d.data.triangle_face = face;
      return face_d;
    }

    ax_device_callable_inlined face_data_s getTransformedFace(const MeshCtx &geometry) const {
      const Object3D &obj3d = geometry.getTriMesh(mesh_id);
      transform4x4_t transform = geometry.getTriMeshTransform(mesh_id);
      geometry::face_data_tri face = triangle::get_face(&obj3d, transform, triangle_id);
      face_data_s face_d{};
      face_d.type = TRIANGLE;
      face_d.data.triangle_face = face;
      return face_d;
    }

    ax_device_callable_inlined transform4x4_t getTransform(const MeshCtx &geometry) const { return geometry.getTriMeshTransform(mesh_id); }

    ax_device_callable_inlined const float *getTransformAddr(const MeshCtx &geometry) const { return geometry.getTriMeshTransformAddr(mesh_id); }

    ax_device_callable_inlined Triangle(std::size_t mesh_id_, std::size_t triangle_id_) {
      AX_ASSERT_LT(mesh_id, uint32_t(-1));
      AX_ASSERT_LT(triangle_id, uint32_t(-1));  // For cuda compatibility
      mesh_id = mesh_id_;
      triangle_id = triangle_id_;
    }

    ax_device_callable_inlined bool hasValidTangents(const MeshCtx &geometry) const {
      geometry::face_data_tri face = getFace(geometry).data.triangle_face;
      return face.hasValidTangents();
    }

    ax_device_callable_inlined bool hasValidBitangents(const MeshCtx &geometry) const {
      geometry::face_data_tri face = getFace(geometry).data.triangle_face;
      return face.hasValidBitangents();
    }

    ax_device_callable_inlined bool hasValidNormals(const MeshCtx &geometry) const {
      geometry::face_data_tri face = getFace(geometry).data.triangle_face;
      return face.hasValidNormals();
    }

    ax_device_callable_inlined bool hasValidUvs(const MeshCtx &geometry) const {
      geometry::face_data_tri face = getFace(geometry).data.triangle_face;
      return face.hasValidUvs();
    }

    ax_device_callable_inlined geometry::BoundingBox computeAABB(const MeshCtx &geometry) const {
      const geometry::face_data_tri face = getFace(geometry).data.triangle_face;
      const transform4x4_t transform = getTransform(geometry);
      vertices_attrb3d_t vertices = face.vertices();
      vertices.v0 = transform.m * glm::vec4(vertices.v0, 1.f);
      vertices.v1 = transform.m * glm::vec4(vertices.v1, 1.f);
      vertices.v2 = transform.m * glm::vec4(vertices.v2, 1.f);

      glm::vec3 min = glm::min(vertices.v0, glm::min(vertices.v1, vertices.v2));
      glm::vec3 max = glm::max(vertices.v0, glm::max(vertices.v1, vertices.v2));
      return {min, max};
    }

    ax_device_callable_inlined glm::vec3 centroid(const MeshCtx &geometry) const {
      geometry::face_data_tri face = getFace(geometry).data.triangle_face;
      transform4x4_t transform = getTransform(geometry);
      return transform.m * glm::vec4(face.compute_center(), 1.f);
    }

    ax_device_callable_inlined bool isDegenerate(const MeshCtx &mesh_geometry) const {
      const Object3D &obj3d = mesh_geometry.getTriMesh(mesh_id);
      const auto &indices = obj3d.indices;
      return indices[triangle_id] == indices[triangle_id + 1] || indices[triangle_id + 2] == indices[triangle_id + 1] ||
             indices[triangle_id + 2] == indices[triangle_id];
    }

    ax_device_callable_inlined float area(const MeshCtx &mesh_geometry) const {
      const geometry::face_data_tri face = getFace(mesh_geometry).data.triangle_face;
      const transform4x4_t transform = getTransform(mesh_geometry);
      const vertices_attrb3d_t vertices = face.vertices();
      glm::vec3 b = vertices.v2 - vertices.v1;
      glm::vec3 s = vertices.v0 - vertices.v1;
      return 0.5f * glm::length(glm::cross(b, s));
    }

    /*
     * https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
     */
    ax_device_callable_inlined bool hit(
        const Ray &in_ray, float tmin, float last_hit_tmax, intersection_record_s &data, const MeshCtx &geometry) const {
      if (isDegenerate(geometry))
        return false;
      const transform4x4_t transform = getTransform(geometry);
      const glm::vec3 origin = transform.inv * glm::vec4(in_ray.origin, 1.f);
      const glm::vec3 direction = transform.inv * glm::vec4(in_ray.direction, 0.f);
      const Ray ray = Ray(origin, direction);
      const geometry::face_data_tri face = getFace(geometry).data.triangle_face;
      const vertices_attrb3d_t vertices = face.vertices();
      const edges_t edges = geometry::compute_edges(vertices);

      const glm::vec3 P = glm::cross(ray.direction, edges.e2);
      const float det = glm::dot(P, edges.e1);
      const float inv_det = 1.f / det;
      if (det == 0.f)
        return false;
      const glm::vec3 T = ray.origin - vertices.v0;
      const float u = glm::dot(P, T) * inv_det;
      if (u < 0 || u > 1.f)
        return false;

      const glm::vec3 Q = glm::cross(T, edges.e1);
      const float v = glm::dot(Q, ray.direction) * inv_det;
      if (v < 0.f || (u + v) > 1.f)
        return false;

      float t = glm::dot(Q, edges.e2) * inv_det;
      if (t < tmin || t > last_hit_tmax)  // Early return in case this triangle is farther than the last intersected shape.
        return false;

      data.geometry.t = t;
      data.geometry.position = transform.m * glm::vec4(ray.pointAt(t), 1.f);
      const float w = 1 - (u + v);
      /* Computes the normals if there aren't any in the vertex attribute buffer. */
      glm::vec3 normal, bitangent, tangent;
      if (!hasValidNormals(geometry)) {
        glm::vec3 computed_normal = glm::cross(edges.e1, edges.e2);
        /* Checks if the eye is on the same hemisphere as the normal. If not , we invert the normal. */
        normal = glm::dot(computed_normal, -ray.direction) < 0 ? -computed_normal : computed_normal;
      } else {
        const vertices_attrb3d_t attrib_normals = face.normals();
        /* Returns barycentric interpolated normal at intersection t.  */
        normal = math::geometry::barycentric_lerp(attrib_normals.v0, attrib_normals.v1, attrib_normals.v2, w, u, v);
      }

      AX_ASSERT_TRUE(hasValidTangents(geometry));
      const vertices_attrb3d_t attrib_tangents = face.tangents();
      tangent = math::geometry::barycentric_lerp(attrib_tangents.v0, attrib_tangents.v1, attrib_tangents.v2, w, u, v);

      const vertices_attrb3d_t attrib_bitangents = face.bitangents();
      if (!hasValidBitangents(geometry)) {
        bitangent = glm::cross(normal, tangent);
      } else {
        bitangent = math::geometry::barycentric_lerp(attrib_bitangents.v0, attrib_bitangents.v1, attrib_bitangents.v2, w, u, v);
      }
      glm::vec3 transformed_normal = transform.n * normal;
      glm::vec3 transformed_tangent = transform.n * tangent;
      data.shading.frame = IntersectFrame(transformed_normal, transformed_tangent);
      data.geometry.wo_dot_n = glm::dot(-in_ray.direction, transformed_normal);
      if (hasValidUvs(geometry)) {
        const vertices_attrb2d_t uvs = face.uvs();
        data.geometry.u = math::geometry::barycentric_lerp(uvs.v0.s, uvs.v1.s, uvs.v2.s, w, u, v);
        data.geometry.v = math::geometry::barycentric_lerp(uvs.v0.t, uvs.v1.t, uvs.v2.t, w, u, v);
      }

      return true;
    }
  };
}  // namespace nova::shape

#endif  // TRIANGLE_H

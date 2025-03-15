#include "Triangle.h"
#include "MeshContext.h"
#include "ray/Ray.h"
#include "shape_datastructures.h"
#include <internal/common/math/utils_3D.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/geometry/Object3D.h>
#include <internal/macro/project_macros.h>

namespace nova::shape {

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
    ax_device_callable_inlined geometry::face_data_tri get_face(const Object3D *geometry,
                                                                const transform::transform4x4_t &transform,
                                                                std::size_t triangle_id) {
      geometry::face_data_tri tri_primitive{};
      const auto &indices = geometry->indices;
      unsigned idx[3] = {indices[triangle_id], indices[triangle_id + 1], indices[triangle_id + 2]};
      geometry->getTri(tri_primitive, idx);
      tri_primitive.transform(transform.m, transform.n);
      return tri_primitive;
    }

  }  // namespace triangle

  ax_device_callable geometry::face_data_tri Triangle::getFace(const MeshCtx &geometry) const {
    const Object3D &obj3d = geometry.getTriMesh(mesh_id);
    geometry::face_data_tri face = triangle::get_face(&obj3d, triangle_id);
    return face;
  }

  ax_device_callable geometry::face_data_tri Triangle::getTransformedFace(const MeshCtx &geometry) const {
    const Object3D &obj3d = geometry.getTriMesh(mesh_id);
    transform::transform4x4_t transform = geometry.getTriMeshTransform(mesh_id);
    geometry::face_data_tri face = triangle::get_face(&obj3d, transform, triangle_id);
    return face;
  }

  ax_device_callable transform::transform4x4_t Triangle::getTransform(const MeshCtx &geometry) const { return geometry.getTriMeshTransform(mesh_id); }

  ax_device_callable Triangle::Triangle(std::size_t mesh_id_, std::size_t triangle_id_) {
    AX_ASSERT_LT(mesh_id, uint32_t(-1));
    AX_ASSERT_LT(triangle_id, uint32_t(-1));  // For cuda compatibility
    mesh_id = mesh_id_;
    triangle_id = triangle_id_;
  }

  ax_device_callable bool Triangle::hasValidTangents(const MeshCtx &geometry) const {
    geometry::face_data_tri face = getFace(geometry);
    return face.hasValidTangents();
  }

  ax_device_callable bool Triangle::hasValidBitangents(const MeshCtx &geometry) const {
    geometry::face_data_tri face = getFace(geometry);
    return face.hasValidBitangents();
  }

  ax_device_callable bool Triangle::hasValidNormals(const MeshCtx &geometry) const {
    geometry::face_data_tri face = getFace(geometry);
    return face.hasValidNormals();
  }

  ax_device_callable ax_no_discard bool Triangle::hasValidUvs(const MeshCtx &geometry) const {
    geometry::face_data_tri face = getFace(geometry);
    return face.hasValidUvs();
  }

  ax_device_callable geometry::BoundingBox Triangle::computeAABB(const MeshCtx &geometry) const {
    const geometry::face_data_tri face = getFace(geometry);
    const transform::transform4x4_t transform = getTransform(geometry);
    vertices_attrb3d_t vertices = face.vertices();
    vertices.v0 = transform.m * glm::vec4(vertices.v0, 1.f);
    vertices.v1 = transform.m * glm::vec4(vertices.v1, 1.f);
    vertices.v2 = transform.m * glm::vec4(vertices.v2, 1.f);

    glm::vec3 min = glm::min(vertices.v0, glm::min(vertices.v1, vertices.v2));
    glm::vec3 max = glm::max(vertices.v0, glm::max(vertices.v1, vertices.v2));
    return {min, max};
  }

  ax_device_callable glm::vec3 Triangle::centroid(const MeshCtx &geometry) const {
    geometry::face_data_tri face = getFace(geometry);
    transform::transform4x4_t transform = getTransform(geometry);
    return transform.m * glm::vec4(face.compute_center(), 1.f);
  }

  ax_device_callable bool Triangle::isDegenerate(const MeshCtx &mesh_geometry) const {
    const Object3D &obj3d = mesh_geometry.getTriMesh(mesh_id);
    const auto &indices = obj3d.indices;
    return indices[triangle_id] == indices[triangle_id + 1] || indices[triangle_id + 2] == indices[triangle_id + 1] ||
           indices[triangle_id + 2] == indices[triangle_id];
  }

  ax_device_callable float Triangle::area(const MeshCtx &mesh_geometry) const {
    const geometry::face_data_tri face = getFace(mesh_geometry);
    const transform::transform4x4_t transform = getTransform(mesh_geometry);
    const vertices_attrb3d_t vertices = face.vertices();
    glm::vec3 b = vertices.v2 - vertices.v1;
    glm::vec3 s = vertices.v0 - vertices.v1;
    return 0.5f * glm::length(glm::cross(b, s));
  }

  /*
   * https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
   */
  ax_device_callable bool Triangle::hit(const Ray &in_ray, float tmin, float last_hit_tmax, hit_data &data, const MeshCtx &geometry) const {
    using namespace math::geometry;
    if (isDegenerate(geometry))
      return false;
    const transform::transform4x4_t transform = getTransform(geometry);
    const glm::vec3 origin = transform.inv * glm::vec4(in_ray.origin, 1.f);
    const glm::vec3 direction = transform.inv * glm::vec4(in_ray.direction, 0.f);
    const Ray ray = Ray(origin, direction);
    const geometry::face_data_tri face = getFace(geometry);
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

    data.t = t;
    data.position = transform.m * glm::vec4(ray.pointAt(data.t), 1.f);
    const float w = 1 - (u + v);
    /* Computes the normals if there aren't any in the vertex attribute buffer. */
    if (!hasValidNormals(geometry)) {
      glm::vec3 computed_normal = glm::cross(edges.e1, edges.e2);
      /* Checks if the eye is on the same hemisphere as the normal. If not , we invert the normal. */
      data.normal = glm::dot(data.normal, -ray.direction) < 0 ? -computed_normal : computed_normal;
    } else {
      const vertices_attrb3d_t normals = face.normals();
      /* Returns barycentric interpolated normal at intersection t.  */
      data.normal = barycentric_lerp(normals.v0, normals.v1, normals.v2, w, u, v);
    }
    data.normal = glm::normalize(transform.n * data.normal);

    if (hasValidUvs(geometry)) {
      const vertices_attrb2d_t uvs = face.uvs();
      data.v = barycentric_lerp(uvs.v0.s, uvs.v1.s, uvs.v2.s, w, u, v);
      data.u = barycentric_lerp(uvs.v0.t, uvs.v1.t, uvs.v2.t, w, u, v);
    }

    const vertices_attrb3d_t tangents = face.tangents();
    data.tangent = barycentric_lerp(tangents.v0, tangents.v1, tangents.v2, w, u, v);
    data.tangent = glm::normalize(transform.n * data.tangent);

    const vertices_attrb3d_t bitangents = face.bitangents();
    if (!hasValidBitangents(geometry)) {
      data.bitangent = glm::normalize(transform.n * glm::cross(data.normal, data.tangent));
    } else {
      data.bitangent = barycentric_lerp(bitangents.v0, bitangents.v1, bitangents.v2, w, u, v);
      data.bitangent = glm::normalize(transform.n * data.bitangent);
    }
    AX_ASSERT_TRUE(hasValidTangents(geometry));

    return true;
  }

}  // namespace nova::shape

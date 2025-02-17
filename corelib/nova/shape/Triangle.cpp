#include "Triangle.h"
#include "ray/Ray.h"
#include "shape_datastructures.h"
#include <internal/common/math/utils_3D.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/device/gpgpu/device_utils.h>
#include <internal/geometry/Object3D.h>
#include <internal/macro/project_macros.h>

namespace nova::shape {

  namespace internals {
    /* We keep track of the mesh list in a static variable to fit multiple Triangle instances into a cache line. */
    static ax_device_managed nova::shape::mesh_shared_views_t geometry_views;
  }  // namespace internals

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

  ax_device_callable geometry::face_data_tri Triangle::getFace() const {
    mesh_properties_t properties = getMesh();
    geometry::face_data_tri face = triangle::get_face(properties.geometry, triangle_id);
    return face;
  }

  ax_device_callable geometry::face_data_tri Triangle::getTransformedFace() const {
    mesh_properties_t properties = getMesh();
    transform::transform4x4_t transform = getTransform();
    geometry::face_data_tri face = triangle::get_face(properties.geometry, transform, triangle_id);
    return face;
  }

  ax_device_callable transform::transform4x4_t Triangle::getTransform() const {
    mesh_properties_t mesh_properties = getMesh();
    transform::transform4x4_t transform{};
    int err = transform::reconstruct_transform4x4(transform, mesh_properties.transform_offset, internals::geometry_views.transforms);
    AX_ASSERT(err == 0, "Problem reconstructing the transformation matrix.");
    return transform;
  }

  ax_host_only void Triangle::updateSharedBuffers(const mesh_shared_views_t &geo) { internals::geometry_views = geo; }

  ax_device_callable mesh_properties_t Triangle::getMesh() const {
    mesh_properties_t prop{};
    prop.transform_offset = transform::get_transform_offset(mesh_id, internals::geometry_views.transforms);
#ifdef __CUDA_ARCH__
    prop.geometry = &(internals::geometry_views.geometry.device_geometry_view)[mesh_id];
    return prop;
#else
    prop.geometry = &(internals::geometry_views.geometry.host_geometry_view)[mesh_id];
    return prop;
#endif
  }

  ax_device_callable Triangle::Triangle(std::size_t mesh_id_, std::size_t triangle_id_) {
    AX_ASSERT_LT(mesh_id, uint32_t(-1));
    AX_ASSERT_LT(triangle_id, uint32_t(-1));  // For cuda compatibility
    mesh_id = mesh_id_;
    triangle_id = triangle_id_;
  }

  ax_device_callable bool Triangle::hasValidTangents() const {
    geometry::face_data_tri face = getFace();
    return face.hasValidTangents();
  }

  ax_device_callable bool Triangle::hasValidBitangents() const {
    geometry::face_data_tri face = getFace();
    return face.hasValidBitangents();
  }

  ax_device_callable bool Triangle::hasValidNormals() const {
    geometry::face_data_tri face = getFace();
    return face.hasValidNormals();
  }

  ax_device_callable ax_no_discard bool Triangle::hasValidUvs() const {
    geometry::face_data_tri face = getFace();
    return face.hasValidUvs();
  }

  ax_device_callable geometry::BoundingBox Triangle::computeAABB() const {
    const geometry::face_data_tri face = getFace();
    const transform::transform4x4_t transform = getTransform();
    vertices_attrb3d_t vertices = face.vertices();
    vertices.v0 = transform.m * glm::vec4(vertices.v0, 1.f);
    vertices.v1 = transform.m * glm::vec4(vertices.v1, 1.f);
    vertices.v2 = transform.m * glm::vec4(vertices.v2, 1.f);

    glm::vec3 min = glm::min(vertices.v0, glm::min(vertices.v1, vertices.v2));
    glm::vec3 max = glm::max(vertices.v0, glm::max(vertices.v1, vertices.v2));
    return {min, max};
  }

  ax_device_callable glm::vec3 Triangle::centroid() const {
    geometry::face_data_tri face = getFace();
    transform::transform4x4_t transform = getTransform();
    return transform.m * glm::vec4(face.compute_center(), 1.f);
  }

  /*
   * https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
   */
  ax_device_callable bool Triangle::hit(const Ray &in_ray, float tmin, float last_hit_tmax, hit_data &data, base_options * /*user_options*/) const {
    using namespace math::geometry;
    const geometry::face_data_tri face = getFace();
    const transform::transform4x4_t transform = getTransform();
    const glm::vec3 origin = transform.inv * glm::vec4(in_ray.origin, 1.f);
    const glm::vec3 direction = transform.inv * glm::vec4(in_ray.direction, 0.f);
    const Ray ray = Ray(origin, direction);
    const vertices_attrb3d_t vertices = face.vertices();
    const edges_t edges = geometry::compute_edges(vertices);
    const vertices_attrb3d_t normals = face.normals();
    const vertices_attrb3d_t tangents = face.tangents();
    const vertices_attrb3d_t bitangents = face.bitangents();
    const vertices_attrb2d_t uvs = face.uvs();

    const glm::vec3 P = glm::cross(ray.direction, edges.e2);
    const float det = glm::dot(P, edges.e1);

    /* backface cull */
    // if (det < epsilon && det > -epsilon)
    //  return false;

    const float inv_det = 1.f / det;
    const glm::vec3 T = ray.origin - vertices.v0;
    const float u = glm::dot(P, T) * inv_det;
    if (u < 0 || u > 1.f)
      return false;

    const glm::vec3 Q = glm::cross(T, edges.e1);
    const float v = glm::dot(Q, ray.direction) * inv_det;
    if (v < 0.f || (u + v) > 1.f)
      return false;

    data.t = glm::dot(Q, edges.e2) * inv_det;
    if (data.t < tmin || data.t > last_hit_tmax)  // Early return in case this triangle is farther than the last intersected shape.
      return false;

    const float w = 1 - (u + v);
    if (!hasValidNormals()) {
      data.normal = glm::cross(edges.e1, edges.e2);
      if (glm::dot(data.normal, -ray.direction) < 0)
        data.normal = -data.normal;
      data.normal = glm::normalize(data.normal);
    } else {
      /* Returns barycentric interpolated normal at intersection t.  */
      data.normal = glm::normalize(barycentric_lerp(normals.v0, normals.v1, normals.v2, w, u, v));
    }
    if (hasValidUvs()) {
      data.v = barycentric_lerp(uvs.v0.s, uvs.v1.s, uvs.v2.s, w, u, v);
      data.u = barycentric_lerp(uvs.v0.t, uvs.v1.t, uvs.v2.t, w, u, v);
    }
    data.tangent = barycentric_lerp(tangents.v0, tangents.v1, tangents.v2, w, u, v);
    data.bitangent = barycentric_lerp(bitangents.v0, bitangents.v1, bitangents.v2, w, u, v);
    AX_ASSERT_TRUE(hasValidBitangents());
    AX_ASSERT_TRUE(hasValidTangents());

    data.position = transform.m * glm::vec4(ray.pointAt(data.t), 1.f);
    data.normal = transform.n * glm::vec4(data.normal, 0.f);
    data.tangent = transform.n * glm::vec4(data.normal, 0.f);
    data.bitangent = transform.n * glm::vec4(data.normal, 0.f);

    return true;
  }

}  // namespace nova::shape

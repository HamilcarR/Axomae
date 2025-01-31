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
    ax_device_callable_inlined geometry::face_data_tri get_face(const nova::shape::mesh_properties_t &mesh_props, std::size_t triangle_id) {
      geometry::face_data_tri tri_primitive{};
      const auto &indices = mesh_props.geometry->indices;
      unsigned idx[3] = {indices[triangle_id], indices[triangle_id + 1], indices[triangle_id + 2]};
      mesh_props.geometry->getTri(tri_primitive, idx);
      return tri_primitive;
    }
  }  // namespace triangle

  ax_device_callable geometry::face_data_tri Triangle::getFace() const {
    mesh_properties_t properties = getMesh();
    geometry::face_data_tri face = triangle::get_face(properties, triangle_id);
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
    geometry::face_data_tri face = getFace();
    vertices_attrb3d_t vertices = face.vertices();
    glm::vec3 min = glm::min(vertices.v0, glm::min(vertices.v1, vertices.v2));
    glm::vec3 max = glm::max(vertices.v0, glm::max(vertices.v1, vertices.v2));
    return {min, max};
  }

  ax_device_callable glm::vec3 Triangle::centroid() const {
    geometry::face_data_tri face = getFace();
    return face.compute_center();
  }

  /*
   * https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
   */
  ax_device_callable bool Triangle::hit(const Ray &ray, float tmin, float tmax, hit_data &data, base_options * /*user_options*/) const {
    using namespace math::geometry;
    mesh_properties_t mesh_props = getMesh();
    transform::transform4x4_t transform = getTransform();
    const glm::vec3 direction = transform.inv * glm::vec4(ray.direction, 0.f);
    const glm::vec3 origin = transform.inv * glm::vec4(ray.origin, 1.f);
    const Ray transformed_ray = Ray(origin, direction);
    geometry::face_data_tri face = getFace();
    vertices_attrb3d_t vertices = face.vertices();
    edges_t edges = geometry::compute_edges(vertices);
    vertices_attrb3d_t normals = face.normals();
    vertices_attrb3d_t tangents = face.tangents();
    vertices_attrb3d_t bitangents = face.bitangents();
    vertices_attrb2d_t uvs = face.uvs();

    glm::vec3 P = glm::cross(transformed_ray.direction, edges.e2);
    const float det = glm::dot(P, edges.e1);
    /* backface cull */
    // if (det < epsilon && det > -epsilon)
    //  return false;

    const float inv_det = 1.f / det;
    glm::vec3 T = transformed_ray.origin - vertices.v0;
    const float u = glm::dot(P, T) * inv_det;
    if (u < 0 || u > 1.f)
      return false;

    glm::vec3 Q = glm::cross(T, edges.e1);
    const float v = glm::dot(Q, transformed_ray.direction) * inv_det;
    if (v < 0.f || (u + v) > 1.f)
      return false;

    data.t = glm::dot(Q, edges.e2) * inv_det;
    /* Early return in case this triangle is farther than the last intersected shape. */
    if (data.t < tmin || data.t > tmax)
      return false;

    const float w = 1 - (u + v);
    if (!hasValidNormals()) {
      data.normal = glm::cross(edges.e1, edges.e2);
      if (glm::dot(data.normal, -transformed_ray.direction) < 0)
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

    return true;
  }

}  // namespace nova::shape

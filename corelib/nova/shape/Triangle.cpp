#include "Triangle.h"
#include "ray/Ray.h"
#include <internal/common/math/utils_3D.h>
#include <internal/device/gpgpu/device_transfer_interface.h>
#include <internal/geometry/Object3D.h>

namespace nova::shape {

  /* We keep track of the mesh list in a static variable to fit multiple Triangle instances into a cache line. */
  static const axstd::span<Object3D> *triangle_mesh_list = nullptr;
#ifdef AXOMAE_USE_CUDA
  ax_device_managed axstd::span<Object3D> device_triangle_mesh_list;
#endif

  ax_host_only void Triangle::updateCpuMeshList(const axstd::span<Object3D> *mesh_list) { triangle_mesh_list = mesh_list; }

  ax_host_only void Triangle::updateGpuMeshList(const axstd::span<Object3D> *mesh_list) {
#ifdef AXOMAE_USE_CUDA
    device_triangle_mesh_list = *mesh_list;
#endif
  }

  ax_device_callable const Object3D *Triangle::getMesh() const {
#ifdef __CUDA_ARCH__
    return &(device_triangle_mesh_list)[mesh_id];
#else
    AX_ASSERT_NOTNULL(triangle_mesh_list);
    return &(*triangle_mesh_list)[mesh_id];
#endif
  }

  ax_device_callable Triangle::Triangle(std::size_t mesh_id_, std::size_t triangle_id_) {
    mesh_id = mesh_id_;
    triangle_id = triangle_id_;
  }

  ax_device_callable Triangle::Triangle(const glm::vec3 vertices[3],
                                        const glm::vec3 normals[3],
                                        const glm::vec2 textures[3],
                                        const glm::vec3 tangents[3],
                                        const glm::vec3 bitangents[3]) {

    bool can_compute_tangents = normals && bitangents;
    bool can_compute_bitangents = normals && tangents;
    bool should_compute_tangents = can_compute_tangents && !tangents;
    bool should_compute_bitangents = can_compute_bitangents && !bitangents;

    if (tangents) {
      t0 = tangents[0];
      t1 = tangents[1];
      t2 = tangents[2];
    }

    if (normals) {
      n0 = normals[0];
      n1 = normals[1];
      n2 = normals[2];
    }
    if (bitangents) {
      b0 = bitangents[0];
      b1 = bitangents[1];
      b2 = bitangents[2];
    }

    if (should_compute_tangents) {
      t0 = glm::normalize(glm::cross(n0, b0));
      t1 = glm::normalize(glm::cross(n1, b1));
      t2 = glm::normalize(glm::cross(n2, b2));
    }

    if (should_compute_bitangents) {
      b0 = glm::normalize(glm::cross(n0, t0));
      b1 = glm::normalize(glm::cross(n1, t1));
      b2 = glm::normalize(glm::cross(n2, t2));
    }

    if (textures) {
      uv0 = textures[0];
      uv1 = textures[1];
      uv2 = textures[2];
      uv_valid = true;
    }

    if (vertices) {
      v0 = vertices[0];
      v1 = vertices[1];
      v2 = vertices[2];
      e1 = v1 - v0;
      e2 = v2 - v0;
      center = (v0 + v1 + v2) * 0.3333f;
    }
  }

  /*
   * https://cadxfem.org/inf/Fast%20MinimumStorage%20RayTriangle%20Intersection.pdf
   */
  ax_device_callable bool Triangle::hit(const Ray &ray, float tmin, float tmax, hit_data &data, base_options *user_options) const {
    using namespace math::geometry;
    glm::vec3 P = glm::cross(ray.direction, e2);
    const float det = glm::dot(P, e1);
    /* backface cull */
    // if (det < epsilon && det > -epsilon)
    //  return false;

    const float inv_det = 1.f / det;
    glm::vec3 T = ray.origin - v0;
    const float u = glm::dot(P, T) * inv_det;
    if (u < 0 || u > 1.f)
      return false;

    glm::vec3 Q = glm::cross(T, e1);
    const float v = glm::dot(Q, ray.direction) * inv_det;
    if (v < 0.f || (u + v) > 1.f)
      return false;

    data.t = glm::dot(Q, e2) * inv_det;
    /* Early return in case this triangle is farther than the last intersected shape. */
    if (data.t < tmin || data.t > tmax)
      return false;

    const float w = 1 - (u + v);
    if (!hasValidNormals()) {
      data.normal = glm::cross(e1, e2);
      if (glm::dot(data.normal, -ray.direction) < 0)
        data.normal = -data.normal;
      data.normal = glm::normalize(data.normal);
    } else {
      /* Returns barycentric interpolated normal at intersection t.  */
      data.normal = glm::normalize(barycentric_lerp(n0, n1, n2, w, u, v));
    }
    if (hasValidUvs()) {
      data.v = barycentric_lerp(uv0.s, uv1.s, uv2.s, w, u, v);
      data.u = barycentric_lerp(uv0.t, uv1.t, uv2.t, w, u, v);
    }
    data.tangent = barycentric_lerp(t0, t1, t2, w, u, v);
    data.bitangent = barycentric_lerp(b0, b1, b2, w, u, v);
    AX_ASSERT_TRUE(hasValidBitangents());
    AX_ASSERT_TRUE(hasValidTangents());

    return true;
  }

  ax_device_callable geometry::BoundingBox Triangle::computeAABB() const {
    glm::vec3 min = glm::min(v0, glm::min(v1, v2));
    glm::vec3 max = glm::max(v0, glm::max(v1, v2));
    return {min, max};
  }
}  // namespace nova::shape

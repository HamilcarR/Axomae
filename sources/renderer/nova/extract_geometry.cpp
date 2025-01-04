#include "Drawable.h"
#include "Mesh.h"
#include "bake.h"
namespace nova_baker_utils {

  static void store_primitive(const Object3D &geometry,
                              const glm::mat4 &final_transfo,
                              const glm::mat3 &normal_matrix,
                              std::size_t alloc_offset_primitives,
                              axstd::span<nova::shape::Triangle> &triangle_buffer,
                              axstd::span<nova::primitive::NovaGeoPrimitive> &primitive_buffer,
                              const nova::material::NovaMaterialInterface &mat,
                              nova::NovaResourceManager &manager,
                              int i) {
    geometry::face_data_tri tri_primitive{};
    unsigned idx[3] = {geometry.indices[i], geometry.indices[i + 1], geometry.indices[i + 2]};
    geometry.getTri(tri_primitive, idx);
    glm::vec3 vertices[3], normals[3], tangents[3], bitangents[3];
    glm::vec2 uv[3];
    geometry::extract_uvs(tri_primitive, uv);
    geometry::transform_tangents(tri_primitive, normal_matrix, tangents);
    geometry::transform_bitangents(tri_primitive, normal_matrix, bitangents);
    geometry::transform_vertices(tri_primitive, final_transfo, vertices);
    geometry::transform_normals(tri_primitive, normal_matrix, normals);
    nova::shape::ShapeResourcesHolder res_holder = manager.getShapeData();
    auto tri = res_holder.add_shape<nova::shape::Triangle>(
        triangle_buffer.data(), alloc_offset_primitives, vertices, normals, uv, tangents, bitangents);
    manager.getPrimitiveData().add_primitive<nova::primitive::NovaGeoPrimitive>(primitive_buffer.data(), alloc_offset_primitives, tri, mat);
  }

  void setup_geometry_data(primitive_buffers_t &geometry_buffers,
                           const drawable_original_transform &drawable_transform,
                           std::size_t &alloc_offset_primitives,
                           nova::material::NovaMaterialInterface &material,
                           nova::NovaResourceManager &manager) {
    glm::mat4 final_transfo = drawable_transform.mesh_original_transformation;
    glm::mat3 normal_matrix = math::geometry::compute_normal_mat(final_transfo);
    const Drawable *drawable = drawable_transform.mesh;
    const Mesh *mesh_object = drawable->getMeshPointer();
    const Object3D &geometry = mesh_object->getGeometry();
    for (int i = 0; i < geometry.indices.size(); i += 3) {
      store_primitive(geometry,
                      final_transfo,
                      normal_matrix,
                      alloc_offset_primitives,
                      geometry_buffers.triangle_alloc_buffer,
                      geometry_buffers.geo_primitive_alloc_buffer,
                      material,
                      manager,
                      i);
      alloc_offset_primitives++;
    }
    nova::shape::ShapeResourcesHolder res_holder = manager.getShapeData();
    res_holder.addTriangleMesh(&geometry);
  }
}  // namespace nova_baker_utils
#include "Mesh.h"
#include "bake.h"

namespace nova_baker_utils {
  void transform_vertices(const geometry::face_data_tri &tri_primitive, const glm::mat4 &final_transfo, glm::vec3 vertices[3]) {
    glm::vec3 v1{tri_primitive.v0[0], tri_primitive.v0[1], tri_primitive.v0[2]};
    glm::vec3 v2{tri_primitive.v1[0], tri_primitive.v1[1], tri_primitive.v1[2]};
    glm::vec3 v3{tri_primitive.v2[0], tri_primitive.v2[1], tri_primitive.v2[2]};

    vertices[0] = final_transfo * glm::vec4(v1, 1.f);
    vertices[1] = final_transfo * glm::vec4(v2, 1.f);
    vertices[2] = final_transfo * glm::vec4(v3, 1.f);
  }

  void transform_normals(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 normals[3]) {

    glm::vec3 n1{tri_primitive.n0[0], tri_primitive.n0[1], tri_primitive.n0[2]};
    glm::vec3 n2{tri_primitive.n1[0], tri_primitive.n1[1], tri_primitive.n1[2]};
    glm::vec3 n3{tri_primitive.n2[0], tri_primitive.n2[1], tri_primitive.n2[2]};

    normals[0] = normal_matrix * n1;
    normals[1] = normal_matrix * n2;
    normals[2] = normal_matrix * n3;
  }

  void transform_tangents(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 tangents[3]) {
    glm::vec3 t0{tri_primitive.tan0[0], tri_primitive.tan0[1], tri_primitive.tan0[2]};
    glm::vec3 t1{tri_primitive.tan1[0], tri_primitive.tan1[1], tri_primitive.tan1[2]};
    glm::vec3 t2{tri_primitive.tan2[0], tri_primitive.tan2[1], tri_primitive.tan2[2]};

    tangents[0] = normal_matrix * t0;
    tangents[1] = normal_matrix * t1;
    tangents[2] = normal_matrix * t2;
  }
  void transform_bitangents(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 bitangents[3]) {
    glm::vec3 b0{tri_primitive.bit0[0], tri_primitive.bit0[1], tri_primitive.bit0[2]};
    glm::vec3 b1{tri_primitive.bit1[0], tri_primitive.bit1[1], tri_primitive.bit1[2]};
    glm::vec3 b2{tri_primitive.bit2[0], tri_primitive.bit2[1], tri_primitive.bit2[2]};

    bitangents[0] = normal_matrix * b0;
    bitangents[1] = normal_matrix * b1;
    bitangents[2] = normal_matrix * b2;
  }

  static void normalize_uv(glm::vec2 &textures) {
    if (textures.s > 1 || textures.s < 0)
      textures.s = textures.s - std::floor(textures.s);
    if (textures.t > 1 || textures.t < 0)
      textures.t = textures.t - std::floor(textures.t);
  }

  void extract_uvs(const geometry::face_data_tri &tri_primitive, glm::vec2 textures[3]) {
    textures[0] = {tri_primitive.uv0[0], tri_primitive.uv0[1]};
    textures[1] = {tri_primitive.uv1[0], tri_primitive.uv1[1]};
    textures[2] = {tri_primitive.uv2[0], tri_primitive.uv2[1]};
    normalize_uv(textures[0]);
    normalize_uv(textures[1]);
    normalize_uv(textures[2]);
  }
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
    geometry.get_tri(tri_primitive, idx);
    glm::vec3 vertices[3], normals[3], tangents[3], bitangents[3];
    glm::vec2 uv[3];
    extract_uvs(tri_primitive, uv);
    transform_tangents(tri_primitive, normal_matrix, tangents);
    transform_bitangents(tri_primitive, normal_matrix, bitangents);
    transform_vertices(tri_primitive, final_transfo, vertices);
    transform_normals(tri_primitive, normal_matrix, normals);
    auto tri = manager.getShapeData().add_shape<nova::shape::Triangle>(
        triangle_buffer.data(), alloc_offset_primitives, vertices, normals, uv, tangents, bitangents);
    manager.getPrimitiveData().add_primitive<nova::primitive::NovaGeoPrimitive>(primitive_buffer.data(), alloc_offset_primitives, tri, mat);
  }

  void setup_geometry_data(primitive_buffers_t &geometry_buffers,
                           Mesh *mesh_object,
                           std::size_t &alloc_offset_primitives,
                           nova::material::NovaMaterialInterface &mat,
                           nova::NovaResourceManager &manager) {
    glm::mat4 final_transfo = mesh_object->computeFinalTransformation();
    glm::mat3 normal_matrix = math::geometry::compute_normal_mat(final_transfo);
    const Object3D &geometry = mesh_object->getGeometry();
    for (int i = 0; i < geometry.indices.size(); i += 3) {
      store_primitive(geometry,
                      final_transfo,
                      normal_matrix,
                      alloc_offset_primitives,
                      geometry_buffers.triangle_alloc_buffer,
                      geometry_buffers.geo_primitive_alloc_buffer,
                      mat,
                      manager,
                      i);
      alloc_offset_primitives++;
    }
  }
}  // namespace nova_baker_utils
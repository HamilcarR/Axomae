#include "Logger.h"
#include "MaterialInterface.h"
#include "Mesh.h"
#include "TextureGroup.h"
#include "bake.h"
#include "manager/NovaResourceManager.h"
#include "material/nova_material.h"
#include "project_macros.h"
#include "texturing/nova_texturing.h"

namespace nova_baker_utils {

  const nova::texturing::ImageTexture *extract_texture(const TextureGroup &tgroup, nova::NovaResourceManager &manager, GenericTexture::TYPE type) {
    const GenericTexture *gltexture = tgroup.getTexturePointer(type);
    if (!gltexture) {
      LOG("Texture lookup in Nova scene initialization has returned null.", LogLevel::WARNING);
      return nullptr;
    }
    const auto *buffer_ptr = gltexture->getData();
    int w = (int)gltexture->getWidth();
    int h = (int)gltexture->getHeight();

    return dynamic_cast<nova::texturing::ImageTexture *>(manager.getTexturesData().add_texture<nova::texturing::ImageTexture>(buffer_ptr, w, h, 4));
  }

  const nova::material::NovaMaterialInterface *extract_materials(const Mesh *mesh, nova::NovaResourceManager &manager) {
    const MaterialInterface *material = mesh->getMaterial();
    if (!material) {
      return nullptr;
    }
    const TextureGroup &texture_group = material->getTextureGroup();
    nova::material::texture_pack tpack{};
    tpack.albedo = extract_texture(texture_group, manager, GenericTexture::DIFFUSE);
    tpack.normalmap = extract_texture(texture_group, manager, GenericTexture::NORMAL);
    tpack.metallic = extract_texture(texture_group, manager, GenericTexture::METALLIC);
    tpack.roughness = extract_texture(texture_group, manager, GenericTexture::ROUGHNESS);
    tpack.ao = extract_texture(texture_group, manager, GenericTexture::AMBIANTOCCLUSION);
    tpack.emissive = extract_texture(texture_group, manager, GenericTexture::EMISSIVE);
    int r = math::random::nrandi(0, 2);
    nova::material::NovaMaterialInterface *mat_ptr = nullptr;
    switch (r) {
      case 0:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaConductorMaterial>(tpack, math::random::nrandf(0.001, 0.5));
        break;
      case 1:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaDielectricMaterial>(tpack, math::random::nrandf(1.5, 2.4));
        break;
      case 2:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaDiffuseMaterial>(tpack);
        break;
      default:
        mat_ptr = manager.getMaterialData().add_material<nova::material::NovaConductorMaterial>(tpack, 0.004);
        break;
    }

    return mat_ptr;
  }

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

  static std::size_t compute_primitive_number(const std::vector<Mesh *> &meshes, int indices_padding = 3) {
    std::size_t acc = 0;
    for (const auto &elem : meshes) {
      const Object3D &geometry = elem->getGeometry();
      for (int i = 0; i < geometry.indices.size(); i += indices_padding)
        acc++;
    }
    return acc;
  }

  template<class U>
  static U *allocate_pool_memory(core::memory::Arena<> &memory_pool, std::size_t number_elements) {
    return memory_pool.construct<U>(number_elements, false);
  }

  void build_scene(const std::vector<Mesh *> &meshes, nova::NovaResourceManager &manager) {
    /* Allocate for triangles */
    std::size_t primitive_number = compute_primitive_number(meshes);
    auto *triangle_buffer = allocate_pool_memory<nova::shape::Triangle>(manager.getMemoryPool(), primitive_number);
    auto *primitive_buffer = allocate_pool_memory<nova::primitive::NovaGeoPrimitive>(manager.getMemoryPool(), primitive_number);

    std::size_t alloc_offset = 0;
    for (const auto &elem : meshes) {
      glm::mat4 final_transfo = elem->computeFinalTransformation();
      glm::mat3 normal_matrix = math::geometry::compute_normal_mat(final_transfo);
      const nova::material::NovaMaterialInterface *mat = extract_materials(elem, manager);
      const Object3D &geometry = elem->getGeometry();
      for (int i = 0; i < geometry.indices.size(); i += 3) {
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
            triangle_buffer, alloc_offset, vertices, normals, uv, tangents, bitangents);
        manager.getPrimitiveData().add_primitive<nova::primitive::NovaGeoPrimitive>(primitive_buffer, alloc_offset, tri, mat);
        alloc_offset++;
      }
    }
  }

  void build_acceleration_structure(nova::NovaResourceManager &manager) {
    const auto *primitive_collection_ptr = &manager.getPrimitiveData().get_primitives();
    manager.getAccelerationData().build(primitive_collection_ptr);
  }
}  // namespace nova_baker_utils

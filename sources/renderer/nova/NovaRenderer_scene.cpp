#include "Logger.h"
#include "Mesh.h"
#include "NovaRenderer.h"
#include "Object3D.h"
#include "manager/NovaResourceManager.h"
#include "material/nova_material.h"
#include "primitive/NovaGeoPrimitive.h"
#include "shape/nova_shape.h"
#include "texturing/ConstantTexture.h"

static const nova::texturing::ImageTexture *extract_texture(const TextureGroup &tgroup, nova::NovaResourceManager *manager, Texture::TYPE type) {
  const Texture *gltexture = tgroup.getTexturePointer(type);
  if (!gltexture) {
    LOG("Texture lookup in Nova scene initialization has returned null.", LogLevel::WARNING);
    return nullptr;
  }
  const auto *buffer_ptr = gltexture->getData();
  int w = (int)gltexture->getWidth();
  int h = (int)gltexture->getHeight();

  return dynamic_cast<const nova::texturing::ImageTexture *>(
      manager->getTexturesData().add_texture<nova::texturing::ImageTexture>(buffer_ptr, w, h, 4));
}

static nova::material::NovaMaterialInterface *extract_materials(const Mesh *mesh, nova::NovaResourceManager *manager) {
  const MaterialInterface *material = mesh->getMaterial();
  if (!material) {
    return nullptr;
  }
  const TextureGroup &texture_group = material->getTextureGroup();
  nova::material::texture_pack tpack{};
  tpack.albedo = extract_texture(texture_group, manager, Texture::DIFFUSE);
  tpack.normalmap = extract_texture(texture_group, manager, Texture::NORMAL);
  tpack.metallic = extract_texture(texture_group, manager, Texture::METALLIC);
  tpack.roughness = extract_texture(texture_group, manager, Texture::ROUGHNESS);
  tpack.ao = extract_texture(texture_group, manager, Texture::AMBIANTOCCLUSION);
  tpack.emissive = extract_texture(texture_group, manager, Texture::EMISSIVE);
  int r = math::random::nrandi(0, 2);
  nova::material::NovaMaterialInterface *mat_ptr = nullptr;
  switch (r) {
    case 0:
      mat_ptr = manager->getMaterialData().add_material<nova::material::NovaConductorMaterial>(tpack, 0.0001);
      break;
    case 1:
      mat_ptr = manager->getMaterialData().add_material<nova::material::NovaDielectricMaterial>(tpack, math::random::nrandf(1.5, 2.4));
      break;
    case 2:
      mat_ptr = manager->getMaterialData().add_material<nova::material::NovaDiffuseMaterial>(tpack);
      break;
    default:
      mat_ptr = manager->getMaterialData().add_material<nova::material::NovaConductorMaterial>(tpack, 0.004);
      break;
  }

  return mat_ptr;
}

static void transform_vertices(const geometry::face_data_tri &tri_primitive, const glm::mat4 &final_transfo, glm::vec3 vertices[3]) {
  glm::vec3 v1{tri_primitive.v0[0], tri_primitive.v0[1], tri_primitive.v0[2]};
  glm::vec3 v2{tri_primitive.v1[0], tri_primitive.v1[1], tri_primitive.v1[2]};
  glm::vec3 v3{tri_primitive.v2[0], tri_primitive.v2[1], tri_primitive.v2[2]};

  vertices[0] = final_transfo * glm::vec4(v1, 1.f);
  vertices[1] = final_transfo * glm::vec4(v2, 1.f);
  vertices[2] = final_transfo * glm::vec4(v3, 1.f);
}

static void transform_normals(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 normals[3]) {

  glm::vec3 n1{tri_primitive.n0[0], tri_primitive.n0[1], tri_primitive.n0[2]};
  glm::vec3 n2{tri_primitive.n1[0], tri_primitive.n1[1], tri_primitive.n1[2]};
  glm::vec3 n3{tri_primitive.n2[0], tri_primitive.n2[1], tri_primitive.n2[2]};

  normals[0] = normal_matrix * n1;
  normals[1] = normal_matrix * n2;
  normals[2] = normal_matrix * n3;
}

static void transform_tangents(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 tangents[3]) {
  glm::vec3 t0{tri_primitive.tan0[0], tri_primitive.tan0[1], tri_primitive.tan0[2]};
  glm::vec3 t1{tri_primitive.tan1[0], tri_primitive.tan1[1], tri_primitive.tan1[2]};
  glm::vec3 t2{tri_primitive.tan2[0], tri_primitive.tan2[1], tri_primitive.tan2[2]};

  tangents[0] = normal_matrix * t0;
  tangents[1] = normal_matrix * t1;
  tangents[2] = normal_matrix * t2;
}
static void transform_bitangents(const geometry::face_data_tri &tri_primitive, const glm::mat3 &normal_matrix, glm::vec3 bitangents[3]) {
  glm::vec3 b0{tri_primitive.bit0[0], tri_primitive.bit0[1], tri_primitive.bit0[2]};
  glm::vec3 b1{tri_primitive.bit1[0], tri_primitive.bit1[1], tri_primitive.bit1[2]};
  glm::vec3 b2{tri_primitive.bit2[0], tri_primitive.bit2[1], tri_primitive.bit2[2]};

  bitangents[0] = normal_matrix * b0;
  bitangents[1] = normal_matrix * b1;
  bitangents[2] = normal_matrix * b2;
}

static void extract_uvs(const geometry::face_data_tri &tri_primitive, glm::vec2 textures[3]) {
  textures[0] = {tri_primitive.uv0[0], tri_primitive.uv0[1]};
  textures[1] = {tri_primitive.uv1[0], tri_primitive.uv1[1]};
  textures[2] = {tri_primitive.uv2[0], tri_primitive.uv2[1]};
}

void NovaRenderer::setNewScene(const SceneChangeData &new_scene) {
  namespace nova_material = nova::material;
  namespace nova_primitive = nova::primitive;
  namespace nova_shape = nova::shape;

  resetToBaseState();
  nova_resource_manager->clearResources();
  for (const auto &elem : new_scene.mesh_list) {
    glm::mat4 final_transfo = elem->computeFinalTransformation();
    glm::mat3 normal_matrix = math::geometry::compute_normal_mat(final_transfo);
    nova_material::NovaMaterialInterface *mat = extract_materials(elem, nova_resource_manager.get());
    const Object3D &geometry = elem->getGeometry();
    for (int i = 0; i < geometry.indices.size(); i += 3) {
      geometry::face_data_tri tri_primitive;
      unsigned idx[3] = {geometry.indices[i], geometry.indices[i + 1], geometry.indices[i + 2]};
      geometry.get_tri(tri_primitive, idx);
      glm::vec3 vertices[3], normals[3], tangents[3], bitangents[3];
      glm::vec2 uv[3];
      extract_uvs(tri_primitive, uv);
      transform_tangents(tri_primitive, normal_matrix, tangents);
      transform_bitangents(tri_primitive, normal_matrix, bitangents);
      transform_vertices(tri_primitive, final_transfo, vertices);
      transform_normals(tri_primitive, normal_matrix, normals);
      auto tri = nova_resource_manager->getShapeData().add_shape<nova_shape::Triangle>(vertices, normals, uv, tangents, bitangents);
      nova_resource_manager->getPrimitiveData().add_primitive<nova_primitive::NovaGeoPrimitive>(tri, mat);
    }
  }
  /* Build acceleration. */
  setProgressStatus("Building BVH structure...");
  const auto *primitive_collection_ptr = &nova_resource_manager->getPrimitiveData().get_primitives();
  nova_resource_manager->getAccelerationData().build(primitive_collection_ptr);
  cancel_render = false;
}

void NovaRenderer::prepSceneChange() {
  cancel_render = true;
  emptyScheduler();
  nova_resource_manager->clearResources();
}

Scene &NovaRenderer::getScene() const { return *scene; }

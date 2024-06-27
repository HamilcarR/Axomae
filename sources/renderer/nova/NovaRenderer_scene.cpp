#include "Mesh.h"
#include "NovaRenderer.h"
#include "Object3D.h"
#include "manager/NovaResourceManager.h"
#include "material/nova_material.h"
#include "primitive/NovaGeoPrimitive.h"
#include "shape/nova_shape.h"
#include "texturing/ConstantTexture.h"

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

static void extract_uvs(const geometry::face_data_tri &tri_primitive, glm::vec2 textures[3]) {
  textures[0] = {tri_primitive.uv0[0], tri_primitive.uv0[1]};
  textures[1] = {tri_primitive.uv1[0], tri_primitive.uv1[1]};
  textures[2] = {tri_primitive.uv2[0], tri_primitive.uv2[1]};
}

static nova::material::NovaMaterialInterface *extract_materials(const Mesh *mesh, nova::NovaResourceManager *manager) {
  const MaterialInterface *material = mesh->getMaterial();
  if (!material)
    return nullptr;
  const TextureGroup &texture_group = material->getTextureGroup();
  Texture *gltexture = texture_group.getTexturePointer(Texture::DIFFUSE);
  if (!gltexture)
    return nullptr;
  const auto *buffer_ptr = gltexture->getData();
  int w = (int)gltexture->getWidth();
  int h = (int)gltexture->getHeight();
  auto *tex = manager->getTexturesData().add_texture<nova::texturing::ImageTexture>(buffer_ptr, w, h, 4);
  return manager->getMaterialData().add_material<nova::material::NovaDielectricMaterial>(tex, 1.7);
}

void NovaRenderer::setNewScene(const SceneChangeData &new_scene) {
  namespace nova_material = nova::material;
  namespace nova_primitive = nova::primitive;
  namespace nova_shape = nova::shape;

  resetToBaseState();
  nova_resource_manager->getPrimitiveData().clear();
  nova_resource_manager->getTexturesData().clear();
  nova_resource_manager->getShapeData().clear();
  nova_resource_manager->getMaterialData().clear();
  for (const auto &elem : new_scene.mesh_list) {
    glm::mat4 final_transfo = elem->computeFinalTransformation();
    glm::mat3 normal_matrix = math::geometry::compute_normal_mat(final_transfo);
    nova_material::NovaMaterialInterface *mat = extract_materials(elem, nova_resource_manager.get());
    const Object3D &geometry = elem->getGeometry();
    for (int i = 0; i < geometry.indices.size(); i += 3) {
      geometry::face_data_tri tri_primitive;
      unsigned idx[3] = {geometry.indices[i], geometry.indices[i + 1], geometry.indices[i + 2]};
      geometry.get_tri(tri_primitive, idx);
      glm::vec3 vertices[3], normals[3];
      glm::vec2 uv[3];
      transform_vertices(tri_primitive, final_transfo, vertices);
      transform_normals(tri_primitive, normal_matrix, normals);
      extract_uvs(tri_primitive, uv);
      auto tri = nova_resource_manager->getShapeData().add_shape<nova_shape::Triangle>(vertices, normals, uv);
      nova_resource_manager->getPrimitiveData().add_primitive<nova_primitive::NovaGeoPrimitive>(tri, mat);
    }
  }

  /* Build acceleration. */
  const auto *primitive_collection_ptr = &nova_resource_manager->getPrimitiveData().get_primitives();
  nova_resource_manager->getAccelerationData().build(primitive_collection_ptr);
}

Scene &NovaRenderer::getScene() const { return *scene; }

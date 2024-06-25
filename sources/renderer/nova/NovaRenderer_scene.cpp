#include "Mesh.h"
#include "NovaRenderer.h"
#include "Object3D.h"
#include "manager/NovaResourceManager.h"
#include "material/nova_material.h"
#include "primitive/NovaGeoPrimitive.h"
#include "shape/nova_shape.h"
#include "texturing/NovaTextures.h"

void NovaRenderer::setNewScene(const SceneChangeData &new_scene) {
  namespace nova_material = nova::material;
  namespace nova_primitive = nova::primitive;
  namespace nova_shape = nova::shape;

  resetToBaseState();
  cancel_render = true;
  syncRenderEngineThreads();
  cancel_render = false;
  nova_resource_manager->getPrimitiveData().clear();

  auto *tex1 = nova_resource_manager->getTexturesData().add_texture<nova::texturing::ConstantTexture>(glm::vec4(1.f, 1.f, 1.f, 1.f));
  auto *mat1 = nova_resource_manager->getMaterialData().add_material<nova_material::NovaConductorMaterial>(tex1, 0.0001f);

  for (const auto &elem : new_scene.mesh_list) {
    glm::mat4 final_transfo = elem->computeFinalTransformation();
    glm::mat3 normal_matrix = glm::transpose(glm::inverse(glm::mat3(final_transfo)));

    const Object3D &geometry = elem->getGeometry();
    for (int i = 0; i < geometry.indices.size(); i += 3) {

      geometry::face_data_tri tri_primitive;
      unsigned idx[3] = {geometry.indices[i], geometry.indices[i + 1], geometry.indices[i + 2]};
      geometry.get_tri(tri_primitive, idx);

      glm::vec3 v1{tri_primitive.v0[0], tri_primitive.v0[1], tri_primitive.v0[2]};
      glm::vec3 v2{tri_primitive.v1[0], tri_primitive.v1[1], tri_primitive.v1[2]};
      glm::vec3 v3{tri_primitive.v2[0], tri_primitive.v2[1], tri_primitive.v2[2]};

      v1 = final_transfo * glm::vec4(v1, 1.f);
      v2 = final_transfo * glm::vec4(v2, 1.f);
      v3 = final_transfo * glm::vec4(v3, 1.f);

      glm::vec3 n1{tri_primitive.n0[0], tri_primitive.n0[1], tri_primitive.n0[2]};
      glm::vec3 n2{tri_primitive.n1[0], tri_primitive.n1[1], tri_primitive.n1[2]};
      glm::vec3 n3{tri_primitive.n2[0], tri_primitive.n2[1], tri_primitive.n2[2]};

      n1 = normal_matrix * n1;
      n2 = normal_matrix * n2;
      n3 = normal_matrix * n3;

      glm::vec3 vertices[3] = {v1, v2, v3};
      glm::vec3 normals[3] = {n1, n2, n3};

      auto tri = nova_resource_manager->getShapeData().add_shape<nova_shape::Triangle>(vertices, normals);
      nova_resource_manager->getPrimitiveData().add_primitive<nova_primitive::NovaGeoPrimitive>(tri, mat1);
    }
  }

  /* Build acceleration. */
  const auto *primitive_collection_ptr = &nova_resource_manager->getPrimitiveData().get_primitives();
  nova_resource_manager->getAccelerationData().build(primitive_collection_ptr);
}

Scene &NovaRenderer::getScene() const { return *scene; }

#include "Mesh.h"
#include "NovaGeoPrimitive.h"
#include "NovaRenderer.h"
#include "Object3D.h"
#include "nova_material.h"
#include "shape/Triangle.h"
void NovaRenderer::setNewScene(const SceneChangeData &new_scene) {
  namespace nova_material = nova::material;
  namespace nova_primitive = nova::primitive;
  namespace nova_shape = nova::shape;

  resetToBaseState();
  cancel_render = true;
  syncRenderEngineThreads();
  cancel_render = false;
  nova_engine_data->scene_data.materials_collection.clear();
  nova_engine_data->scene_data.primitives.clear();
  nova_engine_data->scene_data.shapes.clear();
  std::unique_ptr<nova_material::NovaMaterialInterface> mat1 = std::make_unique<nova_material::NovaDielectricMaterial>(
      glm::vec4(0.6f, 0.5f, 0.4f, 1.f), 1.9f);

  std::unique_ptr<nova_material::NovaMaterialInterface> mat2 = std::make_unique<nova_material::NovaConductorMaterial>(glm::vec4(1.f, 1.f, 1.f, 1.f),
                                                                                                                      0.002f);

  std::unique_ptr<nova_material::NovaMaterialInterface> mat3 = std::make_unique<nova_material::NovaDielectricMaterial>(
      glm::vec4(1.0f, 0.2f, 0.1f, 1.f), 1.5f);

  std::unique_ptr<nova_material::NovaMaterialInterface> mat4 = std::make_unique<nova_material::NovaDielectricMaterial>(
      glm::vec4(0.3f, 0.9f, 0.1f, 1.f), 2.4f);

  nova_engine_data->scene_data.materials_collection.push_back(std::move(mat1));
  nova_engine_data->scene_data.materials_collection.push_back(std::move(mat2));
  nova_engine_data->scene_data.materials_collection.push_back(std::move(mat3));
  nova_engine_data->scene_data.materials_collection.push_back(std::move(mat4));
  auto c1 = nova_engine_data->scene_data.materials_collection[0].get();
  auto c2 = nova_engine_data->scene_data.materials_collection[1].get();
  auto c3 = nova_engine_data->scene_data.materials_collection[2].get();
  auto c4 = nova_engine_data->scene_data.materials_collection[3].get();
  nova::material::NovaMaterialInterface *materials[4] = {c1, c2, c3, c4};
  int x = 0;
  for (const auto &elem : new_scene.mesh_list) {
    glm::mat4 final_transfo = elem->computeFinalTransformation();
    glm::mat3 normal_matrix = glm::transpose(glm::inverse(glm::mat3(final_transfo)));

    x = x > 3 ? 0 : x;
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

      auto tri = nova::shape::NovaShapeInterface::create<nova_shape::Triangle>(vertices, normals);
      nova_engine_data->scene_data.shapes.push_back(std::move(tri));
      auto s1 = nova_engine_data->scene_data.shapes.back().get();
      auto primit = nova::primitive::NovaPrimitiveInterface::create<nova_primitive::NovaGeoPrimitive>(s1, materials[x]);
      nova_engine_data->scene_data.primitives.push_back(std::move(primit));
    }
    x++;
  }

  /* Build accelerator */
  const auto *primitive_collection_ptr = &nova_engine_data->scene_data.primitives;
  nova_engine_data->acceleration_structure.accelerator.build(primitive_collection_ptr);
}

Scene &NovaRenderer::getScene() const { return *scene; }

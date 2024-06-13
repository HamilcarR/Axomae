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
                                                                                                                      0.008f);
  nova_engine_data->scene_data.materials_collection.push_back(std::move(mat1));
  nova_engine_data->scene_data.materials_collection.push_back(std::move(mat2));
  auto c1 = nova_engine_data->scene_data.materials_collection[0].get();
  auto c2 = nova_engine_data->scene_data.materials_collection[1].get();
  for (const auto &elem : new_scene.mesh_list) {
    const Object3D &geometry = elem->getGeometry();
    for (int i = 0; i < geometry.indices.size(); i += 3) {
      glm::vec3 v1{}, v2{}, v3{};
      // V1
      int idx = geometry.indices[i] * 3;
      v1.x = geometry.vertices[idx];
      v1.y = geometry.vertices[idx + 1];
      v1.z = geometry.vertices[idx + 2];
      // V2
      idx = geometry.indices[i + 1] * 3;
      v2.x = geometry.vertices[idx];
      v2.y = geometry.vertices[idx + 1];
      v2.z = geometry.vertices[idx + 2];

      idx = geometry.indices[i + 2] * 3;
      v3.x = geometry.vertices[idx];
      v3.y = geometry.vertices[idx + 1];
      v3.z = geometry.vertices[idx + 2];

      v1 = elem->computeFinalTransformation() * glm::vec4(v1, 1.f);
      v2 = elem->computeFinalTransformation() * glm::vec4(v2, 1.f);
      v3 = elem->computeFinalTransformation() * glm::vec4(v3, 1.f);

      auto tri = nova::shape::NovaShapeInterface::create<nova_shape::Triangle>(v1, v2, v3);
      nova_engine_data->scene_data.shapes.push_back(std::move(tri));
      auto s1 = nova_engine_data->scene_data.shapes.back().get();
      auto primit = nova::primitive::NovaPrimitiveInterface::create<nova_primitive::NovaGeoPrimitive>(s1, c1);
      nova_engine_data->scene_data.primitives.push_back(std::move(primit));
    }
  }

  /* Build accelerator */
  const auto *primitive_collection_ptr = &nova_engine_data->scene_data.primitives;
  nova_engine_data->acceleration_structure.accelerator.build(primitive_collection_ptr);
}

Scene &NovaRenderer::getScene() const { return *scene; }

#include "Logger.h"
#include "Mesh.h"
#include "NovaRenderer.h"
#include "Object3D.h"
#include "bake.h"
#include "manager/NovaResourceManager.h"
#include "material/nova_material.h"
#include "primitive/NovaGeoPrimitive.h"
#include "shape/nova_shape.h"
#include "texturing/ConstantTexture.h"

void NovaRenderer::setNewScene(const SceneChangeData &new_scene) {
  AX_ASSERT_NOTNULL(nova_resource_manager);
  resetToBaseState();
  nova_resource_manager->clearResources();
  nova_baker_utils::build_scene(new_scene.mesh_list, *nova_resource_manager);
  /* Build acceleration. */
  setProgressStatus("Building BVH structure...");
  nova_baker_utils::build_acceleration_structure(*nova_resource_manager);
  cancel_render = false;
}

void NovaRenderer::prepSceneChange() {
  cancel_render = true;
  emptyScheduler();
  nova_resource_manager->clearResources();
}

Scene &NovaRenderer::getScene() const { return *scene; }

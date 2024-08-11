#include "ExceptionHandlerUI.h"
#include "Logger.h"
#include "Mesh.h"
#include "NovaRenderer.h"
#include "Object3D.h"
#include "bake.h"
#include "manager/NovaResourceManager.h"
#include "material/nova_material.h"
#include "primitive/NovaGeoPrimitive.h"
#include "shape/nova_shape.h"

#include <GenericException.h>

void NovaRenderer::setNewScene(const SceneChangeData &new_scene) {
  try {
    AX_ASSERT_NOTNULL(nova_resource_manager);
    resetToBaseState();
    nova_resource_manager->clearResources();
    nova_baker_utils::build_scene(new_scene.mesh_list, *nova_resource_manager);
    /* Build acceleration. */
    setProgressStatus("Building BVH structure...");
    nova_baker_utils::build_acceleration_structure(*nova_resource_manager);
    nova_resource_manager->getEngineData().startRender();
  } catch (const exception::CatastrophicFailureException &e) {
    LOG(e.what(), LogLevel::CRITICAL);
    controller::ExceptionInfoBoxHandler::handle(e);
    nova_resource_manager->clearResources();
    abort();
  }
}

void NovaRenderer::prepSceneChange() {
  nova_resource_manager->getEngineData().stopRender();
  emptyScheduler();
  nova_resource_manager->clearResources();
}

Scene &NovaRenderer::getScene() const { return *scene; }

#include "ExceptionHandlerUI.h"
#include "NovaRenderer.h"
#include "bake.h"
#include "manager/NovaResourceManager.h"
#include <internal/common/exception/GenericException.h>
#include <internal/debug/Logger.h>
#include <internal/geometry/Object3D.h>

void NovaRenderer::setNewScene(const SceneChangeData &new_scene) {
  nova_resource_manager = new_scene.nova_resource_manager;
  AX_ASSERT_NOTNULL(nova_resource_manager);
  resetToBaseState();
  nova_resource_manager->getEngineData().is_rendering = true;
}

void NovaRenderer::prepSceneChange() {
  if (!nova_resource_manager)
    return;
  nova_resource_manager->getEngineData().is_rendering = false;
  emptyScheduler();
  nova_resource_manager->clearResources();
}

Scene &NovaRenderer::getScene() const { return *scene; }

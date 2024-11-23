#include "ExceptionHandlerUI.h"
#include "NovaRenderer.h"
#include "bake.h"
#include "internal/common/exception/GenericException.h"
#include "internal/debug/Logger.h"
#include "internal/geometry/Object3D.h"
#include "manager/NovaResourceManager.h"

void NovaRenderer::setNewScene(const SceneChangeData &new_scene) {
  try {
    AX_ASSERT_NOTNULL(nova_resource_manager);
    resetToBaseState();
    nova_resource_manager->clearResources();
    bake_buffers_storage = nova_baker_utils::build_scene(new_scene.mesh_list, *nova_resource_manager);
    /* Build acceleration. */
    setProgressStatus("Building BVH structure...");
    nova::aggregate::Accelerator accelerator = nova_baker_utils::build_performance_acceleration_structure(nova_resource_manager->getPrimitiveData().get_primitives());
    nova_resource_manager->setAccelerationStructure(accelerator);
    nova_resource_manager->getEngineData().is_rendering = true;
  } catch (const exception::CatastrophicFailureException &e) {
    LOG(e.what(), LogLevel::CRITICAL);
    controller::ExceptionInfoBoxHandler::handle(e);
    nova_resource_manager->clearResources();
    abort();
  }
}

void NovaRenderer::prepSceneChange() {
  nova_resource_manager->getEngineData().is_rendering = false;
  emptyScheduler();
  nova_resource_manager->clearResources();
}

Scene &NovaRenderer::getScene() const { return *scene; }

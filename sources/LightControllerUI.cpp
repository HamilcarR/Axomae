#include "../includes/LightControllerUI.h"
#include "../includes/LightBuilder.h"
#include "../includes/constants.h"

void LightController::connect_all_slots() {
  QObject::connect(ui.button_renderer_lighting_PointLights_add, SIGNAL(pressed()), this, SLOT(addPointLight()));
  QObject::connect(ui.button_renderer_lighting_PointLights_delete, SIGNAL(pressed()), this, SLOT(deletePointLight()));
  QObject::connect(
      ui.button_renderer_lighting_DirectionalLights_add, SIGNAL(pressed()), this, SLOT(addDirectionalLight()));
  QObject::connect(
      ui.button_renderer_lighting_DirectionalLights_delete, SIGNAL(pressed()), this, SLOT(deleteDirectionalLight()));
  QObject::connect(ui.button_renderer_lighting_SpotLights_add, SIGNAL(pressed()), this, SLOT(addSpotLight()));
  QObject::connect(ui.button_renderer_lighting_SpotLights_delete, SIGNAL(pressed()), this, SLOT(deleteSpotLight()));

  QObject::connect(
      &viewer_3d->getRenderer(), &Renderer::sceneModified, [this]() { scene_list_view->updateSceneList(); });
}

void LightController::addPointLight() {
  LightData data = loadFromUi<AbstractLight::POINT>();
  // AbstractLight* light = LightBuilder::createPLight(data);
  // viewer_3d->getRenderer().executeMethod<ADD_ELEMENT_POINTLIGHT>(light);
  viewer_3d->getRenderer().pushEvent<Renderer::ON_LEFT_CLICK>(RENDERER_CALLBACK_ENUM::ADD_ELEMENT_POINTLIGHT, data);
  // scene_list_view->updateSceneList();
}

void LightController::deletePointLight() {}
void LightController::addDirectionalLight() {}
void LightController::deleteDirectionalLight() {}
void LightController::addSpotLight() {}
void LightController::deleteSpotLight() {}

template<AbstractLight::TYPE type>
LightData LightController::loadFromUi() const {
  int red = 0, green = 0, blue = 0;
  float intensity = 0.f;
  NodeItem *selected = nullptr;
  ISceneNode *node = nullptr;
  LightData data;
  if constexpr (type == AbstractLight::POINT) {
    red = ui.hslider_renderer_lighting_PointLights_colors_red->value();
    blue = ui.hslider_renderer_lighting_PointLights_colors_blue->value();
    green = ui.hslider_renderer_lighting_PointLights_colors_green->value();
    intensity = ui.dspinbox_renderer_lighting_PointLights_intensity->value();
    float atten_cste = ui.dspinbox_renderer_lighting_PointLights_Attenuation_constant->value();
    float atten_linear = ui.dspinbox_renderer_lighting_PointLights_Attenuation_linear->value();
    float atten_quad = ui.dspinbox_renderer_lighting_PointLights_Attenuation_quadratic->value();
    ui.spinbox_renderer_lighting_PointLights_colors_red->setValue(red);
    ui.spinbox_renderer_lighting_PointLights_colors_green->setValue(green);
    ui.spinbox_renderer_lighting_PointLights_colors_blue->setValue(blue);
    data.loadAttenuation(atten_cste, atten_linear, atten_quad);
    data.name = std::string("Point-Light");
    data.position = glm::vec3(0, 10, 0);
  }
  data.asPbrColor(red, green, blue);
  data.intensity = intensity * 100;
  if (!scene_list_view->selectedItems().empty())
    selected = static_cast<NodeItem *>(scene_list_view->selectedItems().at(0));
  if (selected)
    node = static_cast<ISceneNode *>(scene_list_view->getSceneNode(selected));
  else {
    NodeItem *root = scene_list_view->getRoot();
    node = static_cast<ISceneNode *>(scene_list_view->getSceneNode(root));
  }
  data.parent = node;
  LOG(node->getName(), LogLevel::INFO);
  return data;
}
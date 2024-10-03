#include "LightControllerUI.h"
#include "Renderer.h"
#include "constants.h"
#include "internal/debug/Logger.h"
#include "light/LightingSystem.h"

// TODO : refactor for cleaner controller system : Implement light control system after free camera

void LightController::connectAllSlots() {
  QObject::connect(ui.button_renderer_lighting_PointLights_add, SIGNAL(pressed()), this, SLOT(slot_add_point_light()));
  QObject::connect(ui.button_renderer_lighting_PointLights_delete, SIGNAL(pressed()), this, SLOT(slot_delete_point_light()));
  QObject::connect(ui.button_renderer_lighting_DirectionalLights_add, SIGNAL(pressed()), this, SLOT(slot_add_directional_light()));
  QObject::connect(ui.button_renderer_lighting_DirectionalLights_delete, SIGNAL(pressed()), this, SLOT(slot_delete_directional_light()));
  QObject::connect(ui.button_renderer_lighting_SpotLights_add, SIGNAL(pressed()), this, SLOT(slot_add_spot_light()));
  QObject::connect(ui.button_renderer_lighting_SpotLights_delete, SIGNAL(pressed()), this, SLOT(slot_delete_spot_light()));
  // QObject::connect(&realtime_viewer->getRenderer(), &Renderer::sceneModified, [this]() { scene_list_view->updateSceneList(); });
}

void LightController::slot_add_point_light() {
  // LightData data = loadFromUi<AbstractLight::POINT>();
  //  realtime_viewer->getRenderer().pushEvent<Renderer::ON_LEFT_CLICK>(RENDERER_CALLBACK_ENUM::ADD_ELEMENT_POINTLIGHT, data);
}

void LightController::slot_delete_point_light() {}
void LightController::slot_add_directional_light() {}
void LightController::slot_delete_directional_light() {}
void LightController::slot_add_spot_light() {}
void LightController::slot_delete_spot_light() {}

template<AbstractLight::TYPE type>
LightData LightController::loadFromUi() const {
  int red = 0, green = 0, blue = 0;
  float intensity = 0.f;
  NodeItem *selected = nullptr;
  datastructure::NodeInterface *node = nullptr;
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
    node = static_cast<datastructure::NodeInterface *>(scene_list_view->getSceneNode(selected));
  else {
    NodeItem *root = scene_list_view->getRoot();
    node = static_cast<datastructure::NodeInterface *>(scene_list_view->getSceneNode(root));
  }
  // data.parent = node;
  LOG(node->getName(), LogLevel::INFO);
  return data;
}

/*
 if (!event_callback_stack[ON_LEFT_CLICK].empty()) {
  const auto to_process = event_callback_stack[ON_LEFT_CLICK].front();
  if (to_process.first == ADD_ELEMENT_POINTLIGHT) {  // TODO : pack this in a class
    glm::mat4 inv_v = glm::inverse(scene_camera->getView());
    glm::mat4 inv_p = glm::inverse(scene_camera->getProjection());
    glm::vec4 w_space = glm::vec4(((float)mouse_state.pos_x * 2.f / (float)screen_size.width) - 1.f,
                                  1.f - (float)mouse_state.pos_y * 2.f / (float)screen_size.height,
                                  1.f,
                                  1.f);

LightData data = std::any_cast<LightData>(to_process.second);
w_space = inv_p * w_space;
w_space = inv_v * w_space;
glm::vec3 position = w_space / w_space.w;
position.z = 0.f;
data.position = glm::inverse(scene_camera->getSceneRotationMatrix()) * glm::vec4(position, 1.f);
LOG(std::string("x:") + std::to_string(data.position.x) + std::string("  y:") + std::to_string(data.position.y) + std::string("  z:") +
        std::to_string(data.position.z),
    LogLevel::INFO);
database::node::store<PointLight>(*resource_database.getNodeDatabase(), false, data);
event_callback_stack[ON_LEFT_CLICK].pop();
emit sceneModified();
}
}
*/
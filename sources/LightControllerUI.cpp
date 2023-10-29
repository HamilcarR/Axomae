#include "../includes/LightControllerUI.h"
#include "../includes/constants.h"
#include "../includes/LightBuilder.h"

void LightController::connect_all_slots(){
    QObject::connect(ui.button_renderer_lighting_PointLights_add , SIGNAL(pressed()) , this , SLOT(addPointLight()));
    QObject::connect(ui.button_renderer_lighting_PointLights_delete , SIGNAL(pressed()) , this , SLOT(deletePointLight()));
    QObject::connect(ui.button_renderer_lighting_DirectionalLights_add , SIGNAL(pressed()) , this , SLOT(addDirectionalLight()));
    QObject::connect(ui.button_renderer_lighting_DirectionalLights_delete , SIGNAL(pressed()) , this , SLOT(deleteDirectionalLight()));
    QObject::connect(ui.button_renderer_lighting_SpotLights_add , SIGNAL(pressed()) , this , SLOT(addSpotLight())); 
    QObject::connect(ui.button_renderer_lighting_SpotLights_delete , SIGNAL(pressed()) , this , SLOT(deleteSpotLight())); 
}

void LightController::addPointLight(){
    if(!scene_list_view->selectedItems().empty()){
        LightData data ;
        int red = ui.hslider_renderer_lighting_PointLights_colors_red->value(); 
        int blue = ui.hslider_renderer_lighting_PointLights_colors_blue->value();
        int green = ui.hslider_renderer_lighting_PointLights_colors_green->value();
        float intensity = ui.dspinbox_renderer_lighting_PointLights_intensity->value();   
        float atten_cste = ui.dspinbox_renderer_lighting_PointLights_Attenuation_constant->value();
        float atten_linear = ui.dspinbox_renderer_lighting_PointLights_Attenuation_linear->value(); 
        float atten_quad = ui.dspinbox_renderer_lighting_PointLights_Attenuation_quadratic->value(); 
        data.asPbrColor(red , green , blue);
        data.intensity = intensity * 100; 
        data.loadAttenuation(atten_cste, atten_linear , atten_quad);
        const NodeItem* selected = static_cast<const NodeItem*>(scene_list_view->selectedItems().at(0));
        if(selected){
            ISceneNode *node = static_cast<ISceneNode*>(scene_list_view->getSceneNode(selected));
            data.position = glm::vec3(0 , 10 , 0); // get click position => transform to world space 
            data.parent = node;  
            AbstractLight* light = LightBuilder::createPLight(data);
            viewer_3d->getRenderer().executeMethod<ADD_ELEMENT_POINTLIGHT>(light);  
            LOG(node->getName() , LogLevel::INFO); 
        }
    }
}

void LightController::deletePointLight(){

}
void LightController::addDirectionalLight(){

}
void LightController::deleteDirectionalLight(){

}
void LightController::addSpotLight(){

} 
void LightController::deleteSpotLight(){

}


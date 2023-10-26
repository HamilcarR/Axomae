#include "../includes/LightBuilder.h"


AbstractLight* LightBuilder::createPLight(const LightData& data , ISceneNode* parent){
    return new PointLight(data , parent); 
}

AbstractLight* LightBuilder::createDLight(const LightData& data , ISceneNode* parent){
   return new DirectionalLight(data , parent);  
}

AbstractLight* LightBuilder::createSLight(const LightData& data , ISceneNode* parent){
   return new SpotLight(data , parent);  
}

AbstractLight* LightBuilder::createALight(const LightData& data , ISceneNode* parent){
   //TODO : implement area lights 
   return nullptr;  
}
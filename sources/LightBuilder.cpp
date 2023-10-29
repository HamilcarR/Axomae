#include "../includes/LightBuilder.h"


AbstractLight* LightBuilder::createPLight(const LightData& data ){
    return new PointLight(data); 
}

AbstractLight* LightBuilder::createDLight(const LightData& data){
   return new DirectionalLight(data);  
}

AbstractLight* LightBuilder::createSLight(const LightData& data){
   return new SpotLight(data);  
}

AbstractLight* LightBuilder::createALight(const LightData& data){
   //TODO : implement area lights 
   return nullptr;  
}
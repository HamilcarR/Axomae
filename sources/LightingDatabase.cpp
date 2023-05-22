#include "../includes/LightingDatabase.h"
#include "../includes/UniformNames.h"



LightingDatabase::LightingDatabase(){
   /* AbstractLight *L1 = new PointLight(glm::vec3(200 , 500 , 0) , glm::vec3(1.f , 1.f , 1.f), glm::vec3(1.f , 0.045 , 0.0075) , 3000.f); 
    AbstractLight *L2 = new PointLight(glm::vec3(0 , -500 , 0) , glm::vec3(0.3f , 0.5f , 0.1f) , glm::vec3(1.f , 0.045 , 0.0075), 3000.f) ; 
    AbstractLight *L3 = new PointLight(glm::vec3(0 , 0 , 500) , glm::vec3(0.8f , 0.1f , 0.2f) , glm::vec3(1.f , 0.045 , 0.0075), 5000.f) ;
    AbstractLight *L4 = new PointLight(glm::vec3(0 , 0 , -500) , glm::vec3(0.f , 0.5f , 0.1f) , glm::vec3(1.f , 0.045 , 0.0075), 7000.f) ;
    AbstractLight *L5 = new PointLight(glm::vec3(500, 0 , 0) , glm::vec3(0.f , 1.f , 1.f) , glm::vec3(1.f , 0.045 , 0.0075), 2000.f) ; 
    AbstractLight *L6 = new PointLight(glm::vec3(-500, 0 , 0) , glm::vec3(1.f , 0.5f , 0.1f) , glm::vec3(1.f , 0.045 , 0.075), 5000.f) ;
    addLight(L1); 
    addLight(L2);
    addLight(L3);  
    addLight(L4);  
    addLight(L5);  
    addLight(L6); */

    
    AbstractLight* L1 = new SpotLight(glm::vec3(500.f , 0.f , 0) , glm::vec3(0.f , 0.f , 0), glm::vec3(1.f , 1.f , 1.f) , 15.f , 0.5f); 
    addLight(L1);  
}

LightingDatabase::~LightingDatabase(){

}

bool LightingDatabase::addLight(AbstractLight* light){
    if(!light)
        return false;
    AbstractLight::TYPE type = light->getType(); 
    light_database[type].push_back(light) ; 
    return true;
}

const std::vector<AbstractLight*>& LightingDatabase::getLightsArrayByType(AbstractLight::TYPE type) {
    return light_database[type] ;
}

void LightingDatabase::eraseLightsArray(AbstractLight::TYPE type){
    auto arr = light_database[type] ; 
    for(AbstractLight* A :  arr)
        delete A ; 
    light_database.erase(type); 
}

void LightingDatabase::clearDatabase(){
    for(auto it = light_database.begin() ; it != light_database.end() ; it++)
        eraseLightsArray(it->first); 
    light_database.clear(); 
}

void LightingDatabase::updateShadersData(AbstractLight::TYPE type , Shader* shader , glm::mat4& modelview){
    std::vector<AbstractLight*> array = light_database[type] ; 
    unsigned int index = 0 ; 
    for(auto it = array.begin() ; it != array.end() ; it++ , index ++  )    
        (*it)->updateShaderData(shader , modelview , index);    
}

void LightingDatabase::updateShadersData(Shader* shader , glm::mat4& modelview){
    for(auto it = light_database.begin() ; it != light_database.end() ; it++){
        unsigned int light_num = it->second.size(); 
        switch(it->first){
            case AbstractLight::DIRECTIONAL:
                shader->setUniform(std::string(uniform_name_uint_lighting_directional_number_name) , light_num);
            break;
            case AbstractLight::POINT:
                shader->setUniform(std::string(uniform_name_uint_lighting_point_number_name) , light_num); 
            break;
            case AbstractLight::SPOT:
                shader->setUniform(std::string(uniform_name_uint_lighting_spot_number_name) , light_num);
            break;

        }
        updateShadersData(it->first , shader , modelview); 
    }
}
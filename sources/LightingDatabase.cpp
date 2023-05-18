#include "../includes/LightingDatabase.h"
#include "../includes/UniformNames.h"



LightingDatabase::LightingDatabase(){
    AbstractLight *L1 = new DirectionalLight(glm::vec3(200 , 0 , 0) , glm::vec3(0.5f , 0 , 0) , 1.f); 
    AbstractLight *L2 = new DirectionalLight(glm::vec3(-200 , 0 , 0) , glm::vec3(0.f , 0 , 0.5f) , 1.f); 
    AbstractLight *L3 = new DirectionalLight(glm::vec3(0 , 200 , 0) , glm::vec3(0.f , 0.5f , 0.f) , 1.f);
    AbstractLight *L4 = new DirectionalLight(glm::vec3(0 , 0 , -200) , glm::vec3(0 , 0.5f , 0.5f) , 1.f); 
    AbstractLight *L5 = new DirectionalLight(glm::vec3(0 , 0 , 200) , glm::vec3(0.7f , 0.5f , 0.4f) , 5.f) ;  
    AbstractLight *L6 = new DirectionalLight(glm::vec3(0 , -200 , 0) , glm::vec3(0.5f , 0.5f , 0) , 1.f) ;
    addLight(L1); 
    addLight(L2); 
    addLight(L3);
    addLight(L4); 
    addLight(L5); 
    addLight(L6);  
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
                shader->setUniform(std::string(uniform_name_int_lighting_directional_number_name) , light_num);
            break;
        }
        updateShadersData(it->first , shader , modelview); 
    }
}
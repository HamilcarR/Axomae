#include "../includes/LightingDatabase.h"
#include "../includes/UniformNames.h"



LightingDatabase::LightingDatabase(){ 
   
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
//TODO: [AX-36] fix invalidated iterator after erase
void LightingDatabase::eraseLightsArray(AbstractLight::TYPE type){
    auto arr = light_database[type] ; 
    for(AbstractLight* A :  arr)
        delete A ; 
    light_database.erase(type); 
}

void LightingDatabase::clearDatabase(){
    for(auto it = light_database.begin() ; it != light_database.end() ; it++){
        auto type_array = light_database[it->first] ;
        for(auto *A : type_array){
            delete A ;     
        }
    }
    light_database.clear(); 
}

void LightingDatabase::updateShadersData(AbstractLight::TYPE type , Shader* shader , glm::mat4& view){
    std::vector<AbstractLight*> array = light_database[type] ; 
    unsigned int index = 0 ;
    for(auto it = array.begin() ; it != array.end() ; it++ , index ++  )    
        (*it)->updateShaderData(shader , view , index);    
}

// First loop is to set up the max number of lights of each type. 
void LightingDatabase::updateShadersData(Shader* shader , glm::mat4& view){
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
        updateShadersData(it->first , shader , view); 
    }
}
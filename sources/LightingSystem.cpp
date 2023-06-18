#include "../includes/LightingSystem.h"
#include "../includes/UniformNames.h"






void AbstractLight::updateShaderData(Shader* shader_program , glm::mat4& modelview , unsigned int index)  {
    if(shader_program){
        std::string struct_name = light_struct_name + std::string("[") + std::to_string(index) + std::string("]."); 
        shader_program->setUniform(struct_name+std::string(uniform_name_vec3_lighting_position) , viewspace_position);  
        shader_program->setUniform(struct_name+std::string(uniform_name_float_lighting_intensity) , intensity);
        shader_program->setUniform(struct_name+std::string(uniform_name_vec3_lighting_specular_color) , specularColor);
        shader_program->setUniform(struct_name+std::string(uniform_name_vec3_lighting_diffuse_color) , diffuseColor); 
        shader_program->setUniform(struct_name+std::string(uniform_name_vec3_lighting_ambient_color) , ambientColor); 
    
    }
}


/*****************************************************************************************************************/

DirectionalLight::DirectionalLight(){
    position = glm::vec3(0.f); 
    type = DIRECTIONAL; 
    intensity = 0.f ; 
    specularColor = glm::vec3(1.f); 
    ambientColor = glm::vec3(1.f); 
    diffuseColor = glm::vec3(1.f);
    light_struct_name = std::string(uniform_name_str_lighting_directional_struct_name) ;  
}


DirectionalLight::DirectionalLight(glm::vec3 _position , glm::vec3 _ambientColor , glm::vec3 _diffuseColor , glm::vec3 _specularColor , float _intensity):DirectionalLight(){
    specularColor = _specularColor ;
    ambientColor = _ambientColor ;  
    diffuseColor = _diffuseColor ; 
    position = _position ; 
    intensity = _intensity ;
    type = DIRECTIONAL ; 

}

DirectionalLight::DirectionalLight(glm::vec3 _position , glm::vec3 color , float _intensity):DirectionalLight(){
    position = _position ; 
    specularColor = color ; 
    diffuseColor = color ; 
    ambientColor = color ;
    intensity = _intensity ; 
}

DirectionalLight::~DirectionalLight(){

}

void DirectionalLight::updateShaderData(Shader* shader , glm::mat4& modelview , unsigned int index) {
    viewspace_position = glm::vec3(modelview * glm::vec4(position , 0.f));
    AbstractLight::updateShaderData(shader , modelview , index); 
}

/*****************************************************************************************************************/

PointLight::PointLight(){
    position = glm::vec3(0.f); 
    type = POINT; 
    intensity = 0.f ; 
    specularColor = glm::vec3(1.f); 
    ambientColor = glm::vec3(1.f); 
    diffuseColor = glm::vec3(1.f); 
    attenuation = glm::vec3(1.f) ;
    light_struct_name = std::string(uniform_name_str_lighting_point_struct_name) ;  
}


PointLight::PointLight(glm::vec3 _position , glm::vec3 _ambientColor , glm::vec3 _diffuseColor , glm::vec3 _specularColor , glm::vec3 _attenuation ,  float _intensity):PointLight(){
    specularColor = _specularColor ;
    ambientColor = _ambientColor ;  
    diffuseColor = _diffuseColor ; 
    position = _position ; 
    intensity = _intensity ;
    type = POINT ;
    attenuation = _attenuation ;  

}

PointLight::PointLight(glm::vec3 _position , glm::vec3 color , glm::vec3 _attenuation , float _intensity):PointLight(){
    position = _position ; 
    specularColor = color ; 
    diffuseColor = color ; 
    ambientColor = color ;
    intensity = _intensity ; 
    attenuation = _attenuation ; 
}

PointLight::~PointLight(){

}

void PointLight::updateShaderData(Shader* shader , glm::mat4& modelview , unsigned int index) {
    viewspace_position = glm::vec3(modelview * glm::vec4(position , 1.f));
    if(shader){
        std::string struct_name = light_struct_name + std::string("[") + std::to_string(index) + std::string("]."); 
        shader->setUniform(struct_name + uniform_name_float_lighting_attenuation_constant , attenuation.x);
        shader->setUniform(struct_name + uniform_name_float_lighting_attenuation_linear , attenuation.y); 
        shader->setUniform(struct_name + uniform_name_float_lighting_attenuation_quadratic , attenuation.z); 
    } 
    AbstractLight::updateShaderData(shader , modelview , index); 
}


/*****************************************************************************************************************/


SpotLight::SpotLight(){
     position = glm::vec3(0.f); 
    type = SPOT;  
    theta = 0.f ; 
    intensity = 0.f ; 
    specularColor = glm::vec3(1.f); 
    ambientColor = glm::vec3(1.f); 
    diffuseColor = glm::vec3(1.f);
    direction = glm::vec3(0.f);  
    light_struct_name = std::string(uniform_name_str_lighting_spot_struct_name) ;  
}

SpotLight::SpotLight(glm::vec3 _position , glm::vec3 _direction , glm::vec3 _color , float _cutoff_angle , float _intensity):SpotLight(){
    position = _position ; 
    specularColor = _color ; 
    diffuseColor = _color ; 
    ambientColor = _color ;
    intensity = _intensity ;
    theta = _cutoff_angle ; 
    direction = _direction ;  
}

SpotLight::SpotLight(glm::vec3 _position , glm::vec3 _direction , glm::vec3 _ambient , glm::vec3 _diffuse , glm::vec3 _specular , float _angle , float _intensity):SpotLight(){
    specularColor = _specular ;
    ambientColor = _ambient ;  
    diffuseColor = _diffuse ; 
    position = _position ; 
    intensity = _intensity ;
    theta = _angle ;
    direction = _direction ; 

}

void SpotLight::updateShaderData(Shader* shader , glm::mat4& modelview , unsigned int index) {
    viewspace_position = glm::vec3(modelview * glm::vec4(position , 1.f));  
    viewspace_direction = glm::vec3(modelview * glm::vec4(direction - position, 0.f) ); 
    float rad_theta = glm::radians(theta);  
    if(shader){
        std::string struct_name = light_struct_name + std::string("[") + std::to_string(index) + std::string("]."); 
        shader->setUniform(struct_name + std::string(uniform_name_vec3_lighting_spot_direction) , viewspace_direction);
        shader->setUniform(struct_name + std::string(uniform_name_float_lighting_spot_theta) , rad_theta); 
    } 
    AbstractLight::updateShaderData(shader , modelview , index); 
}





























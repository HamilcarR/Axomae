#include "../includes/LightingSystem.h"
#include "../includes/UniformNames.h"






void AbstractLight::updateShaderData(Shader* shader_program , glm::mat4& modelview , unsigned int index) const {
    if(shader_program){
        glm::vec3 viewspace_position = glm::vec3(modelview * glm::vec4(position , 1.f)); 
        std::string struct_name = std::string(uniform_name_str_lighting_directional_struct_name) + std::string("[") + std::to_string(index) + std::string("]."); 
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


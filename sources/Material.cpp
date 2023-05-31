#include "../includes/Material.h" 
#include "../includes/UniformNames.h"



Material::Material(){
	dielectric_factor = 0.f ;
	shininess = 100.f ;  
	roughness_factor = 0.f ; 
	transmission_factor = 0.f ; 
	emissive_factor = 1.f ; 
	refractive_index = glm::vec2(1.f , 1.5f) ; 
	shader_program = nullptr; 
}

Material::~Material(){

}

void Material::setRefractiveIndexValue(float n1 , float n2){
	refractive_index = glm::vec2(n1 , n2);
	if(shader_program)
		shader_program->setUniform(uniform_name_vec2_material_refractive_index, refractive_index);  
}

void Material::addTexture(int index , Texture::TYPE type){
	textures_group.addTexture(index , type) ;
}

void Material::bind(){
	textures_group.bind(); 

}

/**
 * Initializes the material properties and sets the corresponding uniform values in the
 * shader program.
 */
void Material::initializeMaterial(){
	textures_group.initializeGlTextureData();
	if(shader_program){
		std::string material = std::string(uniform_name_str_material_struct_name) + std::string("."); 
		shader_program->setUniform(material+uniform_name_vec2_material_refractive_index , refractive_index) ; 
		shader_program->setUniform(material+uniform_name_float_material_dielectric_factor , dielectric_factor) ; 
		shader_program->setUniform(material+uniform_name_float_material_roughness_factor , roughness_factor) ; 
		shader_program->setUniform(material+uniform_name_float_material_transmission_factor , transmission_factor) ; 
		shader_program->setUniform(material+uniform_name_float_material_emissive_factor , emissive_factor) ; 
		shader_program->setUniform(material+uniform_name_float_material_shininess_factor , shininess); 
	}
}

void Material::clean(){
	textures_group.clean(); 
}

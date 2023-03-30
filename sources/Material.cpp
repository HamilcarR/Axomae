#include "../includes/Material.h" 




Material::Material(){

}

Material::~Material(){
	

}

void Material::addTexture(unsigned int index , Texture::TYPE type){
	textures_group.addTexture(index , type) ; 

}

void Material::bind(){
	textures_group.bind(); 

}

void Material::initializeMaterial(){
	textures_group.initializeGlTextureData();
}

void Material::clean(){
	textures_group.clean(); 
}

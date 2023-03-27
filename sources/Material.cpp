#include "../includes/Material.h" 




Material::Material(){

}

Material::~Material(){
	

}

void Material::addTexture(TextureData &data , Texture::TYPE type){
	textures_group.addTexture(data , type) ; 

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

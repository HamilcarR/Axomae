#include "../includes/TextureGroup.h"
#include "../includes/Shader.h"
#include "../includes/ResourceDatabaseManager.h"

TextureGroup::TextureGroup(){
	texture_database = ResourceDatabaseManager::getInstance()->getTextureDatabase(); //TODO : change this finally to reference the texture database directly 
	initialized = false;  
}

TextureGroup::TextureGroup(const TextureGroup& texture_group){
	texture_collection = texture_group.getTextureCollection(); 
	initialized = texture_group.isInitialized();
}

TextureGroup::~TextureGroup(){
}

void TextureGroup::addTexture(int index , Texture::TYPE type){
	texture_collection.push_back(texture_database->get(index)); 
	
}

Texture* TextureGroup::getTexturePointer(Texture::TYPE type){
	for(auto A : texture_collection)
		if(A->getTextureType() == type)
			return A ;
	return nullptr; 
}

bool TextureGroup::containsType(Texture::TYPE type){
	for(auto A : texture_collection)
		if(A->getTextureType() == type)
			return true; 
	return false;
}

void TextureGroup::initializeGlTextureData(Shader* shader){
	for(Texture* A : texture_collection){
		if(!A->isInitialized())
			A->setGlData(shader);
	}	
	initialized = true ; 
}


void TextureGroup::clean(){
	initialized = false ;
	texture_collection.clear();  
}

void TextureGroup::bind(){
	for(Texture* A : texture_collection)
		A->bindTexture(); 

}

void TextureGroup::unbind(){
	for(Texture* A : texture_collection)
		A->unbindTexture(); 
}


/****************************************************************************************************************************/

TextureGroup& TextureGroup::operator=(const TextureGroup& texture_group){
	if(this != &texture_group){
		texture_collection = texture_group.getTextureCollection(); 
		initialized = texture_group.isInitialized();
	}
	return *this;  
}
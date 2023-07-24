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
	texture_collection.push_back(index); 
	
}

Texture* TextureGroup::getTexturePointer(Texture::TYPE type){
	for(int id : texture_collection){
		Texture* A = texture_database->get(id); 
		if( A && A->getTextureType() == type)
			return A ;
	}
	return nullptr; 
}

bool TextureGroup::containsType(Texture::TYPE type){
	for(int id : texture_collection){
		Texture* A = texture_database->get(id); 
		if(A && A->getTextureType() == type)
			return true; 
	}
	return false;
	
}

void TextureGroup::initializeGlTextureData(Shader* shader){
	for(int id : texture_collection){
		Texture* A = texture_database->get(id); 
		if(A && !A->isInitialized())
			A->setGlData(shader);
	}	
	initialized = true ; 
}


void TextureGroup::clean(){
	initialized = false ;
	texture_collection.clear();  
}

void TextureGroup::bind(){
	for(int id : texture_collection){
		Texture* A = texture_database->get(id);
		if(A) 
			A->bindTexture(); 
	}
}

void TextureGroup::unbind(){
	for(int id : texture_collection){
		Texture* A = texture_database->get(id);
		if(A) 
			A->unbindTexture(); 
	}
}


/****************************************************************************************************************************/

TextureGroup& TextureGroup::operator=(const TextureGroup& texture_group){
	if(this != &texture_group){
		texture_collection = texture_group.getTextureCollection(); 
		initialized = texture_group.isInitialized();
	}
	return *this;  
}
#include "../includes/TextureGroup.h"
#include "../includes/Shader.h"

TextureGroup::TextureGroup(){
	texture_database = TextureDatabase::getInstance(); 
}

TextureGroup::~TextureGroup(){
}

void TextureGroup::addTexture(int index , Texture::TYPE type){
	texture_collection.push_back(texture_database->get(index)); 
	
}


void TextureGroup::initializeGlTextureData(Shader* shader){
	for(Texture* A : texture_collection){	
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

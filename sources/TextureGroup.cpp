#include "../includes/TextureGroup.h"


TextureGroup::TextureGroup(){
	texture_database = TextureDatabase::getInstance(); 
}

TextureGroup::~TextureGroup(){
}

void TextureGroup::addTexture(unsigned int index , Texture::TYPE type){
	texture_collection.push_back(texture_database->get(index)); 
	
}


void TextureGroup::initializeGlTextureData(){
	for(Texture* A : texture_collection)
		A->setGlData(); 
	initialized = true ; 
}


void TextureGroup::clean(){
	initialized = false ; 
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

#include "../includes/TextureGroup.h"


TextureGroup::TextureGroup(){

}

TextureGroup::~TextureGroup(){
}

void TextureGroup::addTexture(TextureData &tex , Texture::TYPE type){
	Texture* texture = Texture::constructTexture(tex , type); 
	texture_collection.push_back(texture); 
	
}


void TextureGroup::initializeGlTextureData(){
	for(Texture* A : texture_collection)
		A->setGlData(); 
	initialized = true ; 
}


void TextureGroup::clean(){
	for(Texture* A : texture_collection){
		A->clean(); 
		delete A ;
	}
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

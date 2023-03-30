#include "../includes/TextureDatabase.h"
#include <utility> 

TextureDatabase* TextureDatabase::instance = nullptr ; 




TextureDatabase* TextureDatabase::getInstance(){
	if(instance == nullptr)
		instance = new TextureDatabase(); 
	return instance ; 
}

void TextureDatabase::destroy(){
	if(instance != nullptr)
		delete instance ;
	instance = nullptr ; 
}


void TextureDatabase::clean(){
	for(std::pair<unsigned int , Texture*> A : texture_database){
		Texture* texture = A.second ; 
		texture->clean(); 
		delete texture ; 	
	}
	texture_database.clear(); 
}

TextureDatabase::TextureDatabase(){

}
TextureDatabase::~TextureDatabase(){

}


void TextureDatabase::addTexture(unsigned int index , TextureData *texture , Texture::TYPE type){
	std::pair<unsigned int , Texture*> pair(index , Texture::constructTexture(texture , type));
	texture_database.insert(pair); 
}

Texture* TextureDatabase::get(unsigned int index){
	if(texture_database.find(index) == texture_database.end())
		return nullptr ; 
	else
		return texture_database[index] ; 
}

bool TextureDatabase::contains(unsigned int index){
	return texture_database.find(index) != texture_database.end() ; 

}


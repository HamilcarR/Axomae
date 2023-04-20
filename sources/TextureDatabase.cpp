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
	for(std::pair<const unsigned int , Texture*>& A : texture_database){
		A.second->clean(); 
		delete A.second ; 
		A.second = nullptr ; 
	}
	texture_database.clear();
	
}

TextureDatabase::TextureDatabase(){

}
TextureDatabase::~TextureDatabase(){

}


void TextureDatabase::addTexture(unsigned int index , TextureData *texture , Texture::TYPE type){
	texture_database[index] = TextureFactory::constructTexture(texture , type) ; 
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

std::vector<std::pair<unsigned int, Texture*>>  TextureDatabase::getTexturesByType(Texture::TYPE type) const {
	std::vector <std::pair<unsigned int , Texture*>> type_collection ;  
	for(auto &A : texture_database){
		if(A.second->getTextureType() == type) 
			type_collection.push_back(A); 
	}
	return type_collection ; 
}

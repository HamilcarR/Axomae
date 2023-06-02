#include "../includes/TextureDatabase.h"
#include <utility> 
#include <algorithm>

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


void TextureDatabase::hardCleanse(){
	for(std::pair<const int , Texture*>& A : texture_database){
		A.second->clean(); 
		delete A.second ; 
		A.second = nullptr ; 
	}
	texture_database.clear();	
}

void TextureDatabase::softCleanse(){
	for(std::pair<const int , Texture*>  A : texture_database)
		if(A.first >= 0){
			A.second->clean();
			delete A.second ;
			A.second = nullptr ; 
			texture_database.erase(A.first); 
		}
}

TextureDatabase::TextureDatabase(){

}
TextureDatabase::~TextureDatabase(){

}

int TextureDatabase::addTexture(TextureData *texture , Texture::TYPE type , bool keep){
	int index = 0;
	if(keep){
		index = -1 ; 
		while(texture_database[index] != nullptr)
			index -- ; 
	}
	else
		while(texture_database[index] != nullptr)
			index ++ ; 
	Texture* tex = TextureFactory::constructTexture(texture , type) ; 
	texture_database[index] = tex ; 	
	return index ; 
}

Texture* TextureDatabase::get(int index){
	if(texture_database.find(index) == texture_database.end())
		return nullptr ; 
	else
		return texture_database[index] ; 
}

bool TextureDatabase::contains(int index){
	return texture_database.find(index) != texture_database.end() ; 
}

std::vector<std::pair<int, Texture*>>  TextureDatabase::getTexturesByType(Texture::TYPE type) const {
	std::vector <std::pair<int , Texture*>> type_collection ;  
	for(auto &A : texture_database){
		if(A.second->getTextureType() == type) 
			type_collection.push_back(A); 
	}
	return type_collection ; 
}

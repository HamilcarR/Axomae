#include "../includes/TextureDatabase.h"
#include <utility> 
#include <algorithm>





void TextureDatabase::purge(){
	for(std::pair<const int , Texture*>& A : texture_database){
		A.second->clean(); 
		delete A.second ; 
		A.second = nullptr ; 
	}
	texture_database.clear();	
}

void TextureDatabase::clean(){
	std::vector <std::map<const int , Texture*>::iterator> to_destroy; 
	for(auto it = texture_database.begin() ; it != texture_database.end() ; it++)
		if(it->first >= 0){
			it->second->clean();
			delete it->second ;
			it->second = nullptr ; 
			to_destroy.push_back(it); 
		}
	for(auto it : to_destroy)
		texture_database.erase(it); 
}

TextureDatabase::TextureDatabase(){

}
TextureDatabase::~TextureDatabase(){

}

int TextureDatabase::addTexture(TextureData *texture , Texture::TYPE type , bool keep , bool is_dummy){
	int index = 0;
	Texture* tex = nullptr;
	std::lock_guard<std::mutex> lock(resource_mutex);  
	if(keep || is_dummy){
		index = -1 ; 
		while(texture_database[index] != nullptr){
			if(is_dummy && texture_database[index]->getTextureType() == type)
				return index ;
			index -- ; 
		}
	}
	else
		while(texture_database[index] != nullptr)
			index ++ ;
	if(is_dummy)
		tex = TextureFactory::constructTexture(nullptr , type);
	else 
		tex = TextureFactory::constructTexture(texture , type) ; 
	texture_database[index] = tex ; 	
	return index ; 
}

Texture* TextureDatabase::get(const int index){
	const std::lock_guard lock(resource_mutex);  
	auto it = texture_database.begin(); 
	for(it ; it != texture_database.end() ; it++)
		if(it->first == index)
			return it->second; 
	return nullptr; 

}

bool TextureDatabase::contains(const int index){
	const std::lock_guard lock(resource_mutex); 
	return texture_database.find(index) != texture_database.end() ; 
}

std::vector<std::pair<int, Texture*>>  TextureDatabase::getTexturesByType(Texture::TYPE type){
	std::vector <std::pair<int , Texture*>> type_collection ;  
	const std::lock_guard lock(resource_mutex); 
	for(auto &A : texture_database){
		if(A.second->getTextureType() == type) 
			type_collection.push_back(A); 
	}
	return type_collection ; 
}

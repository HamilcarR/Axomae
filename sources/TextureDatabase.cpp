#include <utility> 
#include <algorithm>
#include "../includes/TextureDatabase.h"
#include "../includes/Mutex.h"





void TextureDatabase::purge(){
	Mutex::Lock lock(mutex);
	for(std::pair<const int , Texture*>& A : texture_database){
		A.second->clean(); 
		delete A.second ; 
		A.second = nullptr ; 
	}
	texture_database.clear();	
}

void TextureDatabase::clean(){
	std::vector <std::map<const int , Texture*>::iterator> to_destroy; 
	Mutex::Lock lock(mutex);
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
/*
* This method add textures to the database according to the "keep" criteria. 
* The database map is sorted using two parts : the negative side from index -1 , going down , and the positive from index 0 going up. 
* The negative side is reserved for special textures that don't get erased between scene changes(dummy textures , screen fbo texture).  
* The positive side is freed between each scene change. 
*
*
*/
int TextureDatabase::addTexture(TextureData *texture , Texture::TYPE type , bool keep , bool is_dummy){
	int index = 0;
	Texture* tex = nullptr;
	Mutex::Lock lock(mutex); 
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
	Mutex::Lock lock(mutex); 
	return texture_database[index]; 
}

bool TextureDatabase::contains(const int index){
	Mutex::Lock lock(mutex); 
	return texture_database.find(index) != texture_database.end() ; 
}

std::vector<std::pair<int, Texture*>>  TextureDatabase::getTexturesByType(Texture::TYPE type){
	std::vector <std::pair<int , Texture*>> type_collection ;  
	Mutex::Lock lock(mutex); 			
	for(auto &A : texture_database){
		if(A.second->getTextureType() == type){ 
			type_collection.push_back(A); 
		}
	}
	return type_collection ; 
}

int TextureDatabase::add(Texture* texture , bool keep ){	
	Mutex::Lock lock(mutex); 
	if(keep){
		int index = -1 ; 
		while(texture_database[index] != nullptr)
			index -- ;	
		texture_database[index] = texture ;
		return index;  
	}
	else{
		int index = 0 ; 
		while(texture_database[index] != nullptr)
			index ++ ;
		texture_database[index] = texture ; 
		return index;  
	}
}

std::pair<int , Texture*> TextureDatabase::contains(const Texture* address){
	Mutex::Lock lock(mutex); 
	for(auto A : texture_database)
		if(A.second == address)
			return A ; 
	return std::pair(0 , nullptr); 
}

bool TextureDatabase::remove(const int index){
	Texture* tex = get(index); 
	if(tex){
		tex->clean();
		texture_database.erase(index); 
		return true;
	}
	return false;
}

bool TextureDatabase::remove(const Texture* address){	
	for(auto it = texture_database.begin() ; it != texture_database.end() ; it++){
		if(it->second == address && address != nullptr){
			it->second->clean(); 
			texture_database.erase(it);
			return true ; 
		}
	}
	return false;
}
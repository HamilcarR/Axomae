#ifndef TEXTUREDATABASE_H
#define TEXTUREDATABASE_H

#include <map>
#include "Texture.h" 
#include "TextureFactory.h"
class TextureDatabase{
public:
	static TextureDatabase* getInstance(); 
 	void destroy(); 
	void clean();
	void addTexture(unsigned int index , TextureData* texture , Texture::TYPE type); 
	Texture* get(unsigned int index);
	bool contains(unsigned int index); 
	static bool isInstanced(){ return instance != nullptr ; } ; 
private:
	TextureDatabase(); 	
	virtual ~TextureDatabase(); 
	TextureDatabase (const TextureDatabase&) = delete;
  	TextureDatabase& operator=(const TextureDatabase&) = delete;

private:
	static TextureDatabase* instance ;
	std::map<unsigned int , Texture*> texture_database; 

}; 








#endif 

#ifndef TEXTUREGROUP_H
#define TEXTUREGROUP_H

#include "Texture.h" 
#include "TextureDatabase.h"

class TextureGroup {
public:
	TextureGroup(); 
	virtual ~TextureGroup(); 
	virtual void addTexture(unsigned int index , Texture::TYPE type);
	virtual void initializeGlTextureData();
	virtual void clean(); 
	virtual void bind(); 
	virtual void unbind();
	bool isInitialized(){return initialized;}; 
private:
	std::vector<Texture*> texture_collection; //Array of Pointers to textures in the texture database 
	bool initialized; 
	TextureDatabase* texture_database ; 
};


#endif

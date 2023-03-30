#ifndef TEXTUREDATABASE_H
#define TEXTUREDATABASE_H

#include <map>
#include "Texture.h" 

class TextureDatabase{
public:
	static TextureDatabase* getInstance(); 
 	static void destroy(); 
	void clean();
	void addTexture(unsigned int index , TextureData* texture , Texture::TYPE type); 
	Texture* get(unsigned int index);
	bool contains(unsigned int index); 
private:
	TextureDatabase(); 	
	virtual ~TextureDatabase(); 
	static TextureDatabase* instance ;
	std::map<unsigned int , Texture*> texture_database; 

}; 








#endif 

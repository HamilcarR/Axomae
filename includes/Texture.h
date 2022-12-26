#ifndef TEXTURE_H
#define TEXTURE_H

#include "constants.h" 
#include "utils_3D.h" 

class Texture{
public:
	Texture(); 
	virtual ~Texture();
	

private:
	const char* name ; 
	unsigned int GL_id ; 
	
};




#endif 

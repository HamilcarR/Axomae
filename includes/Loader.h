#ifndef LOADER_H
#define LOADER_H


#include <cstdlib>
#include <assert.h>
#include <iostream>
#include <memory>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "constants.h"
#include "utils_3D.h" 
#include "Mesh.h"
#include "TextureDatabase.h"

/*3D obj loader*/

namespace axomae{



class Loader{
public:
	static Loader* getInstance(); 	
	static std::vector<Mesh> load(const char* file);   				
	static void close(); 	
		
private: 
	Loader();	
	~Loader();
	static Loader* instance; 
	
	
};































}









#endif

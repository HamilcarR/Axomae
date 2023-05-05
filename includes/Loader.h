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
#include "ShaderDatabase.h"

/*3D loader : Will import , construct , and load meshes into the program . Additionnally , loads textures*/
namespace axomae{

class Loader{
public:
	static Loader* getInstance(); 	
	static std::vector<Mesh*> load(const char* file);   				
	static std::string loadShader(const char* filename); 
	static void close(); 	
	static void loadShaderDatabase(); 
private: 
	Loader();	
	~Loader();
	static Mesh* generateCubeMap(unsigned num_textures , bool is_glb_model) ; 
	static std::pair<unsigned int , std::vector<Mesh*>> loadObjects(const char* filename) ; 
	
private:
	static Loader* instance; 
};







}









#endif

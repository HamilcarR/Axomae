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

/**
 * @file Loader.h
 * 3D Loader class 
 * 
 */


/*3D loader : Will import , construct , and load meshes into the program . Additionnally , loads textures*/
namespace axomae{

/**
 * @brief 3D Loader class , implemented as a singleton
 */
class Loader{
public:
	/**
	 * @brief Get the Instance Loader pointer
	 * 
	 * @return Loader* 
	 */
	static Loader* getInstance(); 	
	/**
	 * @brief Load a .glb file
	 * 
	 * @param file Path of the 3D glb model
	 * @return std::vector<Mesh*> 
	 */
	static std::vector<Mesh*> load(const char* file);   				
	/**
	 * @brief Loads a shader file into an std::string
	 * 
	 * @param filename Path of the shader
	 * @return std::string 
	 */
	static std::string loadShader(const char* filename); 
	/**
	 * @brief Delete instance 
	 * 
	 */
	static void close(); 	
	/**
	 * @brief Build the shaders and put them into the shader database
	 * 
	 */
	static void loadShaderDatabase(); 
private: 
	/**
	 * @brief Construct a new Loader object
	 * 
	 */
	Loader();	
	/**
	 * @brief Destroy the Loader object
	 * 
	 */
	~Loader();
	/**
	 * @brief Build an environment map Mesh.  
	 * 
	 * @param is_glb_model Used to load a glb mesh as cubemap 
	 * @return Mesh* 
	 */
	static Mesh* generateCubeMap(bool is_glb_model) ; 
	/**
	 * @brief Loads all meshes in the GLB file. 
	 * Returns and std::pair<A , std::vector<Mesh*>> , with A being the number of textures in the model
	 * @param filename GLB file path
	 * @return std::pair<unsigned int , std::vector<Mesh*>> 
	 */
	static std::pair<unsigned int , std::vector<Mesh*>> loadObjects(const char* filename) ; 
	
private:
	static Loader* instance; /**<Loader instance*/
};







}









#endif

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
#include "ResourceDatabaseManager.h"

/**
 * @file Loader.h
 * Implements a Loader class that will read mesh data and textures from disk
 * 
 */

namespace axomae{

/**
 * @brief 3D Loader class
 */
class Loader{
public:
	
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
	 * @brief Load a .glb file
	 * 
	 * @param file Path of the 3D glb model
	 * @return std::vector<Mesh*> 
	 */
	std::vector<Mesh*> load(const char* file);   				
	
	/**
	 * @brief Loads a shader file into an std::string
	 * 
	 * @param filename Path of the shader
	 * @return std::string 
	 */
	std::string loadShader(const char* filename); 
	
	/**
	 * @brief Delete instance 
	 * 
	 */
	void close(); 	
	
	/**
	 * @brief Build the shaders and put them into the shader database
	 * 
	 */
	void loadShaderDatabase(); 
	
	/**
	 * @brief Build an environment map Mesh.  
	 * 
	 * @param is_glb_model Used to load a glb mesh as cubemap 
	 * @return Mesh* 
	 */
	Mesh* generateCubeMap(bool is_glb_model) ; 
	/**
	 * @brief Loads all meshes in the GLB file. 
	 * Returns and std::pair<A , std::vector<Mesh*>> , with A being the number of textures in the model
	 * @param filename GLB file path
	 * @return std::pair<unsigned int , std::vector<Mesh*>> 
	 */
	std::pair<unsigned int , std::vector<Mesh*>> loadObjects(const char* filename) ; 
	

protected:
	ResourceDatabaseManager *resource_database;
};







}









#endif

#ifndef SHADERDATABASE_H
#define SHADERDATABASE_H
#include <map> 
#include "utils_3D.h"
#include "Shader.h"
#include "RenderingDatabaseInterface.h"

/**
 * @file ShaderDatabase.h
 * A database containing Shader pointers to avoid duplicating and copying Shaders with the same informations 
 * 
 */


/**
 * @brief ShaderDatabase class implementation
 * 
 */
class ShaderDatabase : public RenderingDatabaseInterface<Shader>{
public:
	/**
	 * @brief Construct a new Shader Database object
	 * 
	 */
	ShaderDatabase(); 
	
	/**
	 * @brief Destroy the Shader Database object
	 * 
	 */
	virtual ~ShaderDatabase() ;
		
	/**
	 * @brief Cleans the whole database , Deletes all shaders . 
	 * 
	 */
	void clean() override ; 

	/**
	 * @brief 
	 * 
	 */
	void purge() override;	
		
	/**
	 * @brief This function constructs a shader and stores it in the shader database if it does not already exist and returns it. 
	 * 
	 * @param vertex_code A string containing the source code for the vertex shader.
	 * @param fragment_code A string containing the source code for the fragment shader. 
	 * @param type The type of shader being added to the shader database. 
	 * 
	 * @return Shader* Pointer to the constructed shader , or the existing one
	 *
	 * @see Shader::TYPE
	 */
	Shader* addShader(const std::string vertex_code , const std::string fragment_code ,const Shader::TYPE type);  
	
	/**
	 * The function checks if a shader type exists in a shader database.
	 * 
	 * @param type The parameter "type" is of type Shader::TYPE, which is an enumerated type representing
	 * different types of shaders (e.g. vertex shader, fragment shader, geometry shader, etc.). It is used
	 * to look up a shader in the shader_database map.
	 * 
	 * @return The function `contains` returns a boolean value indicating whether the `shader_database`
	 * contains a shader of the specified `type`.
	 */
	virtual bool contains(const int type) override ;
	
	/**
	 * This function returns a pointer to a shader object of a given type from a shader database, or
	 * nullptr if it does not exist.
	 * 
	 * @param type The parameter "type" is of type Shader::TYPE, which is an enumerated type representing
	 * different types of shaders (e.g. vertex shader, fragment shader, geometry shader, etc.). It is used
	 * to look up a shader in the shader_database map.
	 * 
	 * @return a pointer to a Shader object of the specified type if it exists in the shader_database map.
	 * If the shader of the specified type does not exist in the map, the function returns a null pointer.
	 */
	Shader* get(const int type) override;

	/**
	 * @brief Recompile the database of shaders
	 * 
	 */
	virtual void recompile() ; 

	/**
	 * @brief Initialize the shaders
	 * 
	 */
	virtual void initializeShaders();  
private:
	


private:
	std::map<Shader::TYPE, Shader*> shader_database ; 	/**<std::map with unique shaders associated with their type*/


}; 


#endif 

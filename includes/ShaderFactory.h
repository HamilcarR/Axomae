#ifndef SHADERFACTORY_H
#define SHADERFACTORY_H
#include "Shader.h"


/**
 * @file ShaderFactory.h
 * File defining the creation system for shaders
 * 
 */

/**
 * @brief Shader factory class definition
 * 
 */
class ShaderFactory{
public:
	
	/**
	 * @brief Construct a new ShaderFactory object
	 * 
	 */
	ShaderFactory(); 
	/**
	 * @brief Destroy the Shader Factory object
	 * 
	 */
	virtual ~ShaderFactory(); 
	
	/**
	 * @brief Constructs a shader of type "type" , using vertex_code and fragment_code
	 * 
	 * @param vertex_code Source code of the vertex shader
	 * @param fragment_code Source code of the fragment shader
	 * @param type Type of the shader we want to create
	 * @return Shader* Created shader
	 * @see Shader::TYPE
	 * @see Shader
	 */
	static Shader* constructShader(std::string vertex_code , std::string fragment_code , Shader::TYPE type); 



}; 






















#endif 

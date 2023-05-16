#ifndef SHADER_H
#define SHADER_H

#include "utils_3D.h" 
#include "Texture.h"
#include "DebugGL.h" 
#include "Camera.h" 

/**
 * @file Shader.h
 * Implements various Shader types , like phong shading , PBRs etc 
 * 
 */

/**
 * @brief shaders classes implementation 
 * 
 */
class Shader{
public:
	/**
	 * @brief Shader types enumeration
	 * 
	 */
	enum TYPE : signed 
	{
		GENERIC = 0 , /**<Undefined shader type*/
		BLINN = 1 ,   /**<Blinn-Phong shader*/
		CUBEMAP = 2 , /**<shader used for displaying the environment map*/
		PBR = 3		  /**<PBR shader type*/
	};
public: 
	
	/**
	 * @brief This is a constructor for a Shader class that sets the type to GENERIC.
	 * 
	 */
	Shader();
	
	/**
	 * @brief This is a constructor function for a Shader class that takes in vertex and fragment shader code as
	 * input.
	 * 
	 * @param vertex_code A string containing the source code for the vertex shader of the shader program.
	 * @param fragment_code The fragment shader code as a string. 
	 */
	Shader(const std::string vertex_code , const std::string fragment_code); 
	
	/**
	 * @brief Destroy the Shader object
	 * 
	 */
	virtual ~Shader();
	/**
	 * @brief This function initializes a shader program by compiling and linking vertex and fragment shaders.
	 *  
	 */
	virtual void initializeShader();

	/**
	 * @brief Binds the shader
	 * 
	 */
	virtual void bind();

	/**
	 * @brief Releases the shader by setting glUseProgram to 0 
	 * 
	 */
	virtual void release();
	/**
	 * @brief Cleans up the shader 
	 *  
	 */
	virtual void clean();

	/**
	 * @brief Set the Shaders source code for compilation
	 * 
	 * @param vs Vertex shader source code
	 * @param fs Fragment shader source code
	 *
	 */
	virtual void setShadersRawText(std::string vs , std::string fs) { fragment_shader_txt = fs ; vertex_shader_txt = vs ; } 		
	
	/**
	 * @brief Set the Scene Camera Pointer
	 * 
	 * @param camera Pointer on the main camera
	 */
	void setSceneCameraPointer(Camera *camera);  
	
	/**
	 * @brief Updates the camera computations
	 * 
	 */
	virtual void updateCamera();
	
	/**
	 * @brief Set the Camera Position vector uniforms in the shader
	 * 
	 */
	void setCameraPositionUniform(); 
	
	/**
	 * @brief Set all matrices uniforms in the shader, like projection , view , model etc
	 * 
	 * @param model The model of the mesh using the shader
	 * @see setAllMatricesUniforms(const glm::mat4& projection , const glm::mat4& view , const glm::mat4& model)
	 */
	void setAllMatricesUniforms(const glm::mat4& model);
	
	/**
	 * @brief Set all necessary matrices uniforms in the shader. 
	 * 
	 * @param projection Projection matrix
	 * @param view View matrix
	 * @param model Model matrix of the mesh
	 * @see setAllMatricesUniforms(const glm::mat4& model)
	 */
	void setAllMatricesUniforms(const glm::mat4& projection , const glm::mat4& view , const glm::mat4& model); 
	/**
	 * @brief Normal Matrix for transforming normal vectors alongside the model: computed from the invert modelview matrix and set from here automatically
	 * @param model Modelview matrix of the mesh
	 */
	void setNormalMatrixUniform(const glm::mat4& model); 
	
	/**
	 * @brief Set the Inverse Model Matrix Uniform object
	 * 
	 * @param model Model matrix of the mesh
	 */
	void setInverseModelMatrixUniform(const glm::mat4& model);
	
	/**
	 * @brief Set the inverse ModelView matrix uniform 
	 * 
	 * @param view View matrix of the camera
	 * @param model Model matrix of the mesh 
	 */
	void setInverseModelViewMatrixUniform(const glm::mat4& view , const glm::mat4& model);  
	/**
	 * @brief Set the Model Matrix Uniform object
	 * 
	 * @param model Model matrix
	 */
	void setModelMatrixUniform(const glm::mat4& model);
	
	/**
	 * @brief Set the ModelViewProjection Matrix uniforms 
	 * 
	 * @param model Model matrix 
	 */
	void setModelViewProjection(const glm::mat4& model) ; 
	/**
	 * @brief Set the Model View Projection uniforms
	 * 
	 * @param projection Projection matrix
	 * @param view View matrix
	 * @param model Model matrix
	 */
	void setModelViewProjection(const glm::mat4& projection , const glm::mat4& view , const glm::mat4& model); 
	
	/**
	 * @brief GL API function wrapper for EnableVertexAttribArray
	 * 
	 * @param att ID of the position enabled in the shader
	 */
	void enableAttributeArray(GLuint att);
	/**
	 * @brief GL API wrapper for glVertexAttribPointer
	 * 
	 * @param location Location ID 
	 * @param type GLenum type of each component in the array
	 * @param offset Unused
	 * @param tuplesize Number of component per vertex attribute
	 * @param stride Byte offset between attributes
	 */
	void setAttributeBuffer(GLuint location , GLenum type , int offset , int tuplesize , int stride = 0 ); 
	
	/**
	 * @brief Set a generic uniform in the shader
	 * 
	 * @tparam T Type of the data
	 * @param name Shader uniform location name
	 * @param value Value to save
	 */
	template<typename T> 
	void setUniform(const char* name , const T value){
		glUseProgram(shader_program) ; 
		int location = glGetUniformLocation(shader_program , name);
		setUniformValue(location , value);
	}
	
protected:
	/**
	 * @brief Set all the Texture Uniforms 
	 * 
	 */
	virtual void setTextureUniforms();
	/**
	 * @brief Set the Uniform Value for an int
	 * 
	 * @param location Uniform location
	 * @param value Value to be set
	 */
	void setUniformValue(int location , const int value);
	
	/**
	 * @brief Set the Uniform Value for a 4x4 matrix
	 * 
	 * @param location Uniform location 
	 * @param value Value to be set
	 */
	void setUniformValue(int location , const glm::mat4& value); 
	
	/**
	 * @brief Set the Uniform Value for 3x3 matrix
	 * 
	 * @param location Uniform location
	 * @param value Value to be set
	 */
	void setUniformValue(int location , const glm::mat3& value);
	
	/**
	 * @brief Set the Uniform Value for a vec4
	 * 
     * @param location Uniform location
	 * @param value Value to be set
	 */
	void setUniformValue(int location , const glm::vec4& value); 
	
	/**
	 * @brief Set the Uniform Value for a vec3
	 * 
	 * @param location Uniform location
	 * @param value Value to be set
	 */
	void setUniformValue(int location , const glm::vec3& value);
	
	/**
	 * @brief Set the Uniform Value for a vec2
	 * 
	 * @param location Uniform location
	 * @param value Value to be set
	 */
	void setUniformValue(int location , const glm::vec2& value);  	
private:
	
	/**
	 * @brief Set every combination of matrices uniforms . View , ModelView , Projection x View , Projection , etc
	 * 
	 * @param projection Projection matrix
	 * @param view View matrix
	 * @param model Model matrix
	 */
	void setModelViewProjectionMatricesUniforms(const glm::mat4& projection , const glm::mat4& view , const glm::mat4& model); 
protected:
	TYPE type ;							/**<Type of the shader*/ 
	unsigned int shader_program; 		/**<Shader program ID*/
	unsigned int fragment_shader ; 		/**<Fragment shader ID*/
	unsigned int vertex_shader ;		/**<Vertex shader ID*/
	std::string fragment_shader_txt ; 	/**<Fragment shader source code*/
	std::string vertex_shader_txt ; 	/**<Vertex shader source code*/
	Camera* camera_pointer; 			/**<Camera pointer*/


};

/***********************************************************************************************************************************************************/
class BlinnPhongShader : public Shader{ 
public:
	BlinnPhongShader(); 
	BlinnPhongShader(const std::string vertex_code, const std::string fragment_code) ; 
	virtual ~BlinnPhongShader(); 	

};


/***********************************************************************************************************************************************************/
class CubeMapShader : public Shader {
public:
	CubeMapShader(); 
	CubeMapShader(const std::string vertex_code , const std::string fragment_code); 
	virtual ~CubeMapShader(); 

};

























#endif 

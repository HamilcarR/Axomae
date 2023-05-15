#ifndef GLBUFFERS_H
#define GLBUFFERS_H
#include "utils_3D.h" 

/**
 * @file GLBuffers.h
 * Wrapper for opengl buffer functions 
 * 
 */


/**
 * @brief Wrapper class for Opengl buffer functions
 * 
 */
class GLBuffers{
public:

	/**
	 * @brief Construct a new GLBuffers object
	 * 
	 */
	GLBuffers();
	
	/**
	 * @brief Construct a new GLBuffers object
	 * 
	 * @param geometry Mesh geometry data pointer
	 * @see axomae::Object3D
	 */
	GLBuffers(const axomae::Object3D *geometry) ; 
	
	/**
	 * @brief Destroy the GLBuffers object
	 * 
	 */
	virtual ~GLBuffers();
	
	/**
	 * @brief Sets the current geometry pointer
	 * 
	 * @param geo Pointer on the current mesh geometry 
	 */
	void setGeometryPointer(const axomae::Object3D *geo){geometry = geo;} ; 
	
	/**
	 * @brief Initialize glGenBuffers for all vertex attributes 
	 * 
	 */
	void initializeBuffers();
	
	/**
	 * @brief Checks if VAOs and VBOs are initialized
	 * 
	 * @return true If the buffers are ready to be used
	 */
	bool isReady(); 
	
	/**
	 * @brief Delete GL buffers and VAOs of the mesh 
	 * 
	 */
	void clean(); 
	
	/**
	 * @brief Binds the vertex array object of the mesh
	 * 
	 */
	void bindVao();
	
	/**
	 * @brief Unbind the vertex array object
	 * 
	 */
	void unbindVao(); 
	
	/**
	 * @brief Binds the vertex buffer object
	 * 
	 */
	void bindVertexBuffer(); 
	
	/**
	 * @brief Binds the normal buffer object
	 * 
	 */
	void bindNormalBuffer(); 
	
	/**
	 * @brief Binds the texture buffer object
	 * 
	 */
	void bindTextureBuffer(); 
	
	/**
	 * @brief Binds the color buffer object
	 * 
	 */
	void bindColorBuffer(); 
	
	/**
	 * @brief Binds the index buffer object
	 * 
	 */
	void bindIndexBuffer();
	
	/**
	 * @brief Binds the tangent buffer object
	 * 
	 */
	void bindTangentBuffer();

	/**
	 * @brief Transfers data to GPU using glBufferData on all vertex buffers 
	 * 
	 */
	void fillBuffers(); 
private:
	GLuint vao ;						/**<VAO ID*/ 
	GLuint vertex_buffer ; 				/**<Vertex buffer ID*/
	GLuint normal_buffer ;				/**<Normal buffer ID*/ 
	GLuint index_buffer ; 				/**<Index buffer ID*/
	GLuint texture_buffer ;				/**<Texture buffer ID*/
	GLuint color_buffer ;				/**<Color buffer ID*/
	GLuint tangent_buffer ; 			/**<Tangent buffer ID*/
	const axomae::Object3D *geometry; 	/**<Pointer to the meshe's geometry*/


}; 







#endif

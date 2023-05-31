#ifndef GLRENDERBUFFER_H
#define GLRENDERBUFFER_H
#include "GLBufferInterface.h"


/**
 * @file GLRenderBuffer.h
 * Wrapper for OpenGL render buffers functions  
 * 
 */

/**
 * @class GLRenderBuffer
 * @brief Provides a wrapper for render buffers 
 * 
 */
class GLRenderBuffer : public GLBufferInterface{
public:

    /**
     * @brief Enumeration providing a shorter name for usual internal format values
     * 
     */
    enum INTERNAL_FORMAT : signed {
        EMPTY = -1 ,
        DEPTH16 = GL_DEPTH_COMPONENT16 , 
        DEPTH24 = GL_DEPTH_COMPONENT24 , 
        DEPTH32 = GL_DEPTH_COMPONENT32 , 
        DEPTH32F = GL_DEPTH_COMPONENT32F ,
        DEPTH24_STENCIL8 = GL_DEPTH24_STENCIL8 , 
        DEPTH32F_STENCIL8 = GL_DEPTH32F_STENCIL8 , 
    };

    /**
     * @brief Construct a new GLRenderBuffer object
     * 
     */
    GLRenderBuffer(); 

    /**
     * @brief Construct a new GLRenderBuffer object
     * 
     * @param width Width of the buffer
     * @param height Height of the buffer
     * @param type Internal depth format
     * @see INTERNAL_FORMAT
     */
    GLRenderBuffer(unsigned int width , unsigned int height , INTERNAL_FORMAT type);  
    
    /**
     * @brief Destroy the GLRenderBuffer object
     * 
     */
    virtual ~GLRenderBuffer();
    
    /**
     * @brief Creates a render buffer ID 
     * 
     */
    virtual void initializeBuffers() ;
    
    /**
     * @brief Checks if buffers IDs have been initialized
     * 
     * @return true If ID is ready to use
     */
    virtual bool isReady() const ;

    /**
     * @brief Binds the render buffer for offscreen rendering 
     */
    virtual void bind() ; 

    /**
     * @brief Unbinds the current render buffer 
     */
    virtual void unbind()  ;  

    /**
     * @brief Free the IDs used by OpenGL 
     */
    virtual void clean() ;  

    /**
     * @brief Returns the opengl ID of the RBO 
     *       
     * @return unsigned int ID of the RBO  
     */
    unsigned int getID(){return renderbuffer_id; }

    /**
     * @brief Get the internal format value
     * 
     * @return INTERNAL_FORMAT 
     */
    INTERNAL_FORMAT getFormat(){return format;}

    /**
     * @brief Resizes the render buffer textures
     * 
     * @param width New width 
     * @param height New height
     */
    virtual void resize(unsigned width , unsigned height);

private:
     /**
     * @brief Empty method, as the data for the buffer will be uploaded automatically
     */
    virtual void fillBuffers() ;

protected: 
    unsigned int renderbuffer_id;       /**<ID of the render buffer*/ 
    unsigned int width ;                /**<Width of the buffer*/
    unsigned int height ;               /**<Height of the buffer*/
    INTERNAL_FORMAT format ;            /**<Depth and stencil format value*/


};


#endif
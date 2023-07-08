#ifndef GLFRAMEBUFFER_H
#define GLFRAMEBUFFER_H

#include "GLBufferInterface.h"
#include "GLRenderBuffer.h"
#include "Texture.h"

class GLFrameBuffer : public GLBufferInterface{
public:

    enum INTERNAL_FORMAT : signed {
        EMPTY = -1, 
        COLOR0 = GL_COLOR_ATTACHMENT0 ,
        COLOR1 = GL_COLOR_ATTACHMENT1 , 
        COLOR2 = GL_COLOR_ATTACHMENT2 , 
        COLOR3 = GL_COLOR_ATTACHMENT3 , 
        COLOR4 = GL_COLOR_ATTACHMENT4 , 
        COLOR5 = GL_COLOR_ATTACHMENT5 , 
        COLOR6 = GL_COLOR_ATTACHMENT6 ,
        COLOR7 = GL_COLOR_ATTACHMENT7 , 
        DEPTH = GL_DEPTH_ATTACHMENT , 
        STENCIL = GL_STENCIL_ATTACHMENT , 
        DEPTH_STENCIL = GL_DEPTH_STENCIL_ATTACHMENT
    };

    /**
     * @brief Construct a new GLFrameBuffer object
     * 
     */
    GLFrameBuffer(); 

    /**
     * @brief Construct a new GLFrameBuffer with a render buffer attached
     * 
     * @param width Width to pass to GLRenderBuffer renderbuffer_objet variable creation  
     * @param height Height to pass to GLRenderBuffer renderbuffer_object variable creation
     * @param texture_id The created texture ID to attach to the framebuffer 
     * @param renderbuffer_internal_format Depth format to use for the attached GLRenderBuffer
     * @param default_fbo_id_pointer A pointer on the ID of the default framebuffer . In the case of only one context , will be nullptr . 
     */
    GLFrameBuffer(unsigned width , unsigned height , unsigned texture_id , INTERNAL_FORMAT format = COLOR0, GLRenderBuffer::INTERNAL_FORMAT renderbuffer_internal_format = GLRenderBuffer::EMPTY , unsigned int* default_fbo_id_pointer = nullptr); 
    
    /**
     * @brief Destroy the GLFrameBuffer object
     * 
     */
    virtual ~GLFrameBuffer(); 

    /**
     * @brief Generates the framebuffer's ID. 
     * !Note : This method should be called after the framebuffer texture generation as it uses : glFramebufferTexture2D
     * 
     */
    virtual void initializeBuffers() override ; 

    /**
     * @brief Checks if framebuffer is ready to use 
     * 
     */
    virtual bool isReady() const override; 

    /**
     * @brief Binds the framebuffer 
     * 
     */
    virtual void bind() override ; 

    /**
     * @brief Unbinds the framebuffer
     * 
     */
    virtual void unbind() override ; 

    /**
     * @brief Frees all ressources allocated by the framebuffer
     * 
     */
    virtual void clean() override ;

    /**
     * @brief Resize the textures of the framebuffer and render buffer
     * 
     * @param width New width 
     * @param height  New height
     */
    virtual void resize(unsigned int width , unsigned int height) ; 
private:
   
     /**
     * @brief Dummy method 
     * 
     */
    virtual void fillBuffers() override ; 

protected:
    INTERNAL_FORMAT format;                     /*<Internal Format*/ 
    GLRenderBuffer *renderbuffer_object ;       /*<Pointer on the renderbuffer */ 
    unsigned int framebuffer_id ;               /*<Framebuffer ID*/ 
    unsigned int texture_id ;                   /*<ID of the texture rendered to*/
    unsigned int *pointer_on_default_fbo_id ;   /*<Pointer on default fbo variable*/ 










};

















#endif 
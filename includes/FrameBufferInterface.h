#ifndef FRAMEBUFFERINTERFACE_H
#define FRAMEBUFFERINTERFACE_H

#include "Texture.h"
#include "ShaderDatabase.h"
#include "Drawable.h"
#include "GLFrameBuffer.h"
#include "Mesh.h"


/**
 * @file FrameBufferInterface.h
 * This file implements an interface for framebuffers 
 * 
 */


/**
 * @class FrameBufferInterface 
 * This class implements a generic framebuffer
 */
class FrameBufferInterface{
public: 

    /**
     * @brief Construct a new Frame Buffer Interface object
     * 
     */
    FrameBufferInterface(); 

    /**
     * @brief Construct a new Frame Buffer Interface object
     * 
     * @param texture_database Pointer on the texture database 
     * @param texture_size Pointer on the ScreenSize property from the Renderer 
     */
    FrameBufferInterface(TextureDatabase* texture_database , ScreenSize* texture_size); 
    
    /**
     * @brief Destroy the Frame Buffer Interface object
     * 
     */
    virtual ~FrameBufferInterface(); 

    /**
     * @brief Resizes the textures used by the framebuffer . 
     * Will use the values stored inside the texture_dim pointer property
     * 
     */
    virtual void resize();

    /**
     * @brief Set new screen dimensions
     * @param pointer_on_texture_size Pointer on screen  
     * 
     */
    virtual void setTextureDimensions(ScreenSize* pointer_on_texture_size){texture_dim = pointer_on_texture_size;}

    /**
     * @brief Render to the texture stored inside the framebuffer 
     * 
     */
    virtual void bindFrameBuffer() ; 

    /**
     * @brief Unbind the framebuffer , and use the default framebuffer.  
     * 
     */
    virtual void unbindFrameBuffer() ; 

    /**
     * @brief Initializes shaders , and render buffers textures . 
     * 
     */
    virtual void initializeFrameBuffer();
    
    /**
     * @brief Clean the framebuffer's data  
     * 
     */
    virtual void clean();

    
protected:
    int setUpEmptyTexture(TextureDatabase* database , unsigned width , unsigned height); 
    
    void initializeFrameBufferTexture(); 
protected:
    GLFrameBuffer *gl_framebuffer_object;  
    ScreenSize *texture_dim ;
    TextureDatabase* texture_database;
    Texture* fbo_texture_pointer; 
    unsigned int *default_framebuffer_pointer;
    int texture_id ;            /**<ID of the framebuffer texture in the Texture database*/

};





#endif
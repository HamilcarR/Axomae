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
    FrameBufferInterface(Texture::TYPE rendertype = Texture::FRAMEBUFFER); 

    /**
     * @brief Construct a new Frame Buffer Interface object
     * 
     * @param texture_database 
     * @param texture_size 
     * @param default_fbo_id_pointer
     * @param color_attachment 
     * @param rendertype 
     * @param internal_format 
     * @param data_format 
     * @param data_type 
     */
    FrameBufferInterface(TextureDatabase* texture_database , 
                        ScreenSize* texture_size , 
                        unsigned int* default_fbo_id_pointer = nullptr ,
                        GLFrameBuffer::INTERNAL_FORMAT color_attachment = GLFrameBuffer::COLOR0 , 
                        Texture::TYPE rendertype = Texture::FRAMEBUFFER , 
                        Texture::FORMAT internal_format = Texture::RGBA ,
                        Texture::FORMAT data_format = Texture::BGRA ,
                        Texture::FORMAT data_type = Texture::UBYTE 
                        ); 
    
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
     * @param pointer_on_texture_size Pointer on texture size 
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

    /**
     * @brief Set the Default Frame Buffer Id Pointer object
     * 
     * @param id 
     */
    virtual void setDefaultFrameBufferIdPointer(unsigned *id){default_framebuffer_pointer = id ; }


    Texture* getFrameBufferTexturePointer() const {return fbo_texture_pointer;}

    /**
     * @brief Initialize an empty target texture to be rendered to , and returns it's database ID 
     * 
     * @param database The texture database.
     * @param width Width of the target texture
     * @param height Height of the target texture 
     * @param type Type of the target texture , can be of type Texture::FRAMEBUFFER , or Texture::CUBEMAP
     * @return int Database ID of this texture
     */
    virtual int setUpEmptyTexture(unsigned width , unsigned height , Texture::TYPE type); 
    
    /**
     * @brief Initializes an empty texture on a framebuffer. 
     * The resulting texture will be stored in the texture database , and saves it in "fbo_texture_pointer" property 
     * 
     */
    virtual void initializeFrameBufferTexture(); 

protected:
    Texture::TYPE render_type ;
    Texture::FORMAT internal_format ;
    GLFrameBuffer::INTERNAL_FORMAT color_attachment ; 
    Texture::FORMAT data_format ; 
    Texture::FORMAT data_type ; 
    GLFrameBuffer *gl_framebuffer_object;  
    ScreenSize *texture_dim ;
    TextureDatabase* texture_database;
    Texture* fbo_texture_pointer; 
    unsigned int *default_framebuffer_pointer; //! use as argument in constructor , or getters and setters
    int texture_id ;            /**<ID of the framebuffer texture in the Texture database*/
};





#endif
#ifndef FRAMEBUFFERINTERFACE_H
#define FRAMEBUFFERINTERFACE_H

#include "Texture.h"
#include "ShaderDatabase.h"
#include "Drawable.h"
#include "GLFrameBuffer.h"
#include "Mesh.h"





class FrameBufferInterface{
public: 

    FrameBufferInterface(); 

    FrameBufferInterface(TextureDatabase* texture_database , ScreenSize* texture_size); 
    virtual ~FrameBufferInterface(); 


    virtual void resize(); 
    /**
     * @brief Set the Screen Dimensions
     * 
     * @param width 
     * @param height 
     */
    virtual void setTextureDimensions(ScreenSize* pointer_on_texture_size){texture_dim = pointer_on_texture_size;}

    virtual void bindFrameBuffer() ; 

    virtual void unbindFrameBuffer() ; 

    virtual void initializeFrameBuffer();
    
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
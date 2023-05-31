#ifndef CAMERAFRAMEBUFFER_H
#define CAMERAFRAMEBUFFER_H

#include "Texture.h"
#include "ShaderDatabase.h"
#include "Drawable.h"
#include "GLFrameBuffer.h"
#include "Mesh.h"

class CameraFrameBuffer {
public:
   
    /**
     * @brief Construct a new Camera Frame Buffer object
     * 
     * @param texture_database
     * @param shader_database
     * @param width 
     * @param height 
     */
    CameraFrameBuffer(TextureDatabase* texture_database ,ShaderDatabase* shader_database,  ScreenSize* screen_size_pointer , unsigned int *default_fbo_id);  
    
    /**
     * @brief Destroy the Camera Frame Buffer object
     * 
     */
    virtual ~CameraFrameBuffer(); 
    
    /**
     * @brief Set the Screen Dimensions
     * 
     * @param width 
     * @param height 
     */
    virtual void setScreenDimensions(ScreenSize* pointer_on_screen_size){screen_dim = pointer_on_screen_size;;}

    virtual void resize();

    /**
     * @brief 
     * 
     */
    virtual void bindFrameBuffer() ; 

    virtual void unbindFrameBuffer() ; 

    virtual void renderFrameBufferMesh(); 

    Drawable* getDrawable(){return drawable_screen_quad;}

    virtual void startDraw();

    virtual void initializeFrameBuffer(); 

    virtual void clean(); 
private:
    CameraFrameBuffer(); 


protected: 
    Drawable *drawable_screen_quad; 
    GLFrameBuffer *gl_framebuffer_object;  
    ScreenSize *screen_dim ; 
private:
    axomae::Mesh *mesh_screen_quad;
    TextureDatabase* texture_database;
    Texture* fbo_texture_pointer; 
    ShaderDatabase* shader_database;  
    int screen_texture_database_id ;            /**<ID of the framebuffer texture in the Texture database*/
    Shader* shader_framebuffer;
    unsigned int *default_framebuffer_pointer;  
}; 















#endif
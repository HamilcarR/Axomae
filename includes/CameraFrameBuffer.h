#ifndef CAMERAFRAMEBUFFER_H
#define CAMERAFRAMEBUFFER_H

#include "FrameBufferInterface.h"

class CameraFrameBuffer : public FrameBufferInterface{
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
     
    virtual void renderFrameBufferMesh(); 

    Drawable* getDrawable(){return drawable_screen_quad;}
    
    virtual void initializeFrameBuffer() override; 

    virtual void startDraw();

    virtual void clean() override;

    virtual void updateFrameBufferShader();

private:
    CameraFrameBuffer(); 


protected: 
    Drawable *drawable_screen_quad; 
    axomae::Mesh *mesh_screen_quad;
    ShaderDatabase* shader_database;  
    Shader* shader_framebuffer;
    float gamma ;
    float exposure ;  
    
}; 















#endif